# klein_policies.jl — the 13 sorting policies of Table 1 (Klein 1999) as
# described in the paper, used as tie-breakers over start_time (the paper:
# "sorts tasks in increasing order based on their possible start times. In
# cases where tasks share identical start times, it employs one of the
# policies presented below").
#
# Implemented as one concrete callable struct (branch on mode) so sort!
# specializes once instead of dynamically dispatching per comparison.
#
# Underspecifications in the paper, resolved here (documented choices):
#   - HWRU's ω has no value in the paper           -> ω = 0.5
#   - EFTD/STFD "dynamic" values                   -> use the candidate's
#     current start_time (which already reflects resource availability)
#   - initial policy pick (Algorithm 1 getPolicy)  -> first of the list (SAD)

struct KleinData
    est :: Vector{Float64}
    eft :: Vector{Float64}
    lst :: Vector{Float64}
    lft :: Vector{Float64}
    fullsucc :: Vector{Set{Int64}}
end

function kleinData(inputs)
    n = inputs.nTasks
    est = zeros(n); eft = zeros(n)
    # PSPLIB activity ids are topologically ordered (predecessor id < task id)
    for task in inputs.tasks
        e = 0.0
        for (pid, _) in task.predecessors
            e = max(e, eft[pid])
        end
        est[task.id] = e
        eft[task.id] = e + task.duration
    end
    lft = zeros(n); lst = zeros(n)
    for task in reverse(inputs.tasks)
        l = isempty(task.successors) ? eft[n] : minimum(lst[sid] for (sid, _) in task.successors)
        lft[task.id] = l
        lst[task.id] = l - task.duration
    end
    fullsucc = Vector{Set{Int64}}(undef, n)
    for task in inputs.tasks
        vis = Set{Int64}()
        computeSuccessors(inputs.tasks, task, vis)
        delete!(vis, task.id)
        fullsucc[task.id] = vis
    end
    return KleinData(est, eft, lst, lft, fullsucc)
end

const HWRU_OMEGA = 0.5  # ω for HWRU — value not given in the paper

struct KleinPolicy
    mode :: Int8                  # 1=SAD 2=MIS 3=MTS 4=HRPW 5=HRPW* 6=EFT
                                  # 7=EFTD 8=LST 9=LFT 10=STFS 11=STFD 12=HRD 13=HWRU
    d :: KleinData
    tasks :: Vector{Task}
    caps :: Vector{Int64}         # resource capacities indexed by resource id
end

@inline function kleinValue(p::KleinPolicy, t)
    m = p.mode
    m == Int8(1)  && return Float64(t.duration)                                                            # SAD (min)
    m == Int8(2)  && return Float64(length(t.successors))                                                  # MIS (max)
    m == Int8(3)  && return Float64(length(p.d.fullsucc[t.id]))                                            # MTS (max)
    if m == Int8(4)                                                                                        # HRPW (max)
        s = t.duration
        for (sid, _) in t.successors
            s += p.tasks[sid].duration
        end
        return Float64(s)
    end
    if m == Int8(5)                                                                                        # HRPW* (max)
        s = t.duration
        for sid in p.d.fullsucc[t.id]
            s += p.tasks[sid].duration
        end
        return Float64(s)
    end
    m == Int8(6)  && return p.d.eft[t.id]                                                                  # EFT (min)
    m == Int8(7)  && return Float64(t.start_time + t.duration)                                             # EFTD (min, dynamic)
    m == Int8(8)  && return p.d.lst[t.id]                                                                  # LST (min)
    m == Int8(9)  && return p.d.lft[t.id]                                                                  # LFT (min)
    m == Int8(10) && return p.d.lst[t.id] - p.d.est[t.id]                                                  # STFS (min)
    m == Int8(11) && return p.d.lst[t.id] - t.start_time                                                   # STFD (min, dynamic)
    if m == Int8(12)                                                                                       # HRD (max)
        s = 0
        for (_, u) in t.resources
            s += u
        end
        return Float64(t.duration * s)
    end
    # HWRU (max)
    s = 0.0
    for (r, u) in t.resources
        s += u / p.caps[r]
    end
    return HWRU_OMEGA * length(t.successors) + (1 + HWRU_OMEGA) * s
end

# max-policies: MIS, MTS, HRPW, HRPW*, HRD, HWRU
@inline kleinMaximize(mode::Int8) = mode == Int8(2) || mode == Int8(3) || mode == Int8(4) ||
                                    mode == Int8(5) || mode == Int8(12) || mode == Int8(13)

@inline function (p::KleinPolicy)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    va = kleinValue(p, a)
    vb = kleinValue(p, b)
    return kleinMaximize(p.mode) ? va > vb : va < vb
end

function makeKleinPolicies(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    return [KleinPolicy(Int8(m), d, inputs.tasks, caps) for m in 1:13]
end
