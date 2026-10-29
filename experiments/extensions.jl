
struct LearnedPolicy
    pr :: Vector{Float64}          # indexed by task id; updated in place
end
@inline function (p::LearnedPolicy)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return p.pr[a.id] < p.pr[b.id]
end
mutable struct Elite
    k :: Int
    st :: Vector{Vector{Int}}      # start times by task id (snapshots: Tasks are aliased)
    mk :: Vector{Int}
    hs :: Vector{UInt64}
end
_snap(sol, n) = (v = zeros(Int, n); for t in sol.tasks; v[t.id] = t.start_time; end; v)
function _refresh!(p::LearnedPolicy, e::Elite)
    fill!(p.pr, 0.0)
    for v in e.st
        p.pr .+= v
    end
    p.pr ./= length(e.st)
    return nothing
end
function _offer!(e::Elite, v::Vector{Int}, mk::Int)
    h = hash(v)
    h in e.hs && return false
    if length(e.mk) < e.k
        push!(e.st, v); push!(e.mk, mk); push!(e.hs, h)
        return true
    end
    w = argmax(e.mk)
    mk < e.mk[w] || return false
    e.st[w] = v; e.mk[w] = mk; e.hs[w] = h
    return true
end
function ado_e1_start(name::String, seed::Int, tieseed::Int, k::Int)
    mk = Int[]
    n = 0
    e = Elite(k, Vector{Int}[], Int[], UInt64[])
    cd(ADO_SOLVER_ROOT) do
        for j in 1:13
            inp = readInputs(name)
            n = maximum(t.id for t in inp.tasks)
            r = Xoshiro(seed)
            pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
            push!(mk, Int(s.bestSolution.makespan))
            _offer!(e, _snap(s.bestSolution, n), Int(s.bestSolution.makespan))
        end
    end
    learned = LearnedPolicy(zeros(n))
    _refresh!(learned, e)
    pr14 = copy(learned.pr)                 # priorities used by the rule-14 init build
    cd(ADO_SOLVER_ROOT) do
        inp = readInputs(name)
        r = Xoshiro(seed)
        pols, _ = buildPolicies(inp, r, "B2")
        p14 = Any[pols...; learned]
        s = startRun(inp, p14, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = learned)
        @assert s.schedules == 3
        push!(mk, Int(s.bestSolution.makespan))
        _offer!(e, _snap(s.bestSolution, n), Int(s.bestSolution.makespan)) && _refresh!(learned, e)
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        p14 = Any[pols...; learned]
        # rebuild the winner's init schedule; rule 14 must use the priorities it had then
        ip = w == 14 ? LearnedPolicy(pr14) : p14[w]
        st = startRun(inputs, p14, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = ip)
        w == 14 && (st.bestPolicy = learned)
        st
    end
    @assert s.bestSolution.makespan == m
    s.schedules = 42
    s.traj[1] = (iter = 0, schedules = 42, makespan = m, time = 0.0)
    return s, w, length(winners), mk, learned, e, n
end
function ado_e1_run!(s, target::Int, learned::LearnedPolicy, e::Elite, n::Int, share::Float64)
    tries = 0; tries14 = 0; ref14 = 0; rec14 = 0; entered = 0
    klein = s.sortPolicies[1:13]
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = if rand(s.rng, Float64) < share
            learned
        else
            rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, klein)
        end
        is14 = pol === learned
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        tries += 1; tries14 += is14; ref14 += (s.bestPolicy === learned)
        if _offer!(e, _snap(sol, n), Int(sol.makespan))
            _refresh!(learned, e); entered += 1
        end
        if sol.makespan < s.bestSolution.makespan
            s.bestBeta = beta
            is14 || (s.bestPolicy = pol)   # the Klein channel keeps its own reference rule
            s.bestSolution = sol
            rec14 += is14
            push!(s.traj, (iter = s.iter, schedules = s.schedules,
                           makespan = sol.makespan, time = 0.0))
        end
    end
    return (tries, tries14, ref14, rec14, entered, s.bestPolicy === learned)
end


_order(v::Vector{Int}) = sortperm(v)      # ids by start time (stable: ties by id)
function ado_prs_start(name::String, seed::Int, tieseed::Int, k::Int)
    mk = Int[]; n = 0
    e = Elite(k, Vector{Int}[], Int[], UInt64[])
    cd(ADO_SOLVER_ROOT) do
        for j in 1:13
            inp = readInputs(name)
            n = maximum(t.id for t in inp.tasks)
            r = Xoshiro(seed)
            pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
            push!(mk, Int(s.bestSolution.makespan))
            _offer!(e, _snap(s.bestSolution, n), Int(s.bestSolution.makespan))
        end
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[w])
    end
    @assert s.bestSolution.makespan == m
    s.schedules = 39
    s.traj[1] = (iter = 0, schedules = 39, makespan = m, time = 0.0)
    return s, w, e, n
end
function _relink!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, norig::Int, target::Int)
    idx = sortperm(e.mk)
    length(idx) < 2 && return (0, 0)
    guide = _order(e.st[idx[1]])
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    used = 0; found = 0
    for o in idx[2:min(end, norig + 1)]
        order = _order(e.st[o])
        for pos in 1:length(order)
            (used >= cap || s.schedules >= target) && return (used, found)
            order[pos] == guide[pos] && continue
            src = findfirst(==(guide[pos]), order)
            order[pos], order[src] = order[src], order[pos]
            repairPrecedence!(order, s.inputs)
            c0 = SCHEDULE_COUNT[]
            mk = decodeOrder!(s.inputs, s.availability_units, order, ft)
            s.schedules += SCHEDULE_COUNT[] - c0
            used += 1
            st = zeros(Int, n)
            for t in s.inputs.tasks
                st[t.id] = ft[t.id] - t.duration
            end
            _offer!(e, st, mk)
            if mk < gbest[]
                gbest[] = mk; found += 1
                push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
            end
        end
    end
    return (used, found)
end
function ado_prs_run!(s, target::Int, mode::Int, w::Int, e::Elite, n::Int,
                      S::Int, cap::Int, norig::Int, seed2::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    runbest = gbest[]; m0 = gbest[]
    since = 0; episodes = 0; prused = 0; prfound = 0; restarts = 0
    pols = s.sortPolicies
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, pols)
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        s.iter += 1
        mk = Int(sol.makespan)
        mode == 1 && _offer!(e, _snap(sol, n), mk)
        if mk < runbest
            runbest = mk
            s.bestBeta = beta
            s.bestPolicy = pol
        end
        if mk < gbest[]
            gbest[] = mk; since = 0
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        if mode != 0 && since >= S && s.schedules < target
            episodes += 1
            if mode == 1
                u, f = _relink!(s, e, n, gbest, cap, norig, target)
                prused += u; prfound += f
                runbest = min(runbest, gbest[])
            else
                restarts += 1
                s.rng = Xoshiro(seed2 + 7919 * restarts)
                s.bestPolicy = pols[w]
                runbest = m0
            end
            since = 0
        end
    end
    return (s.iter, episodes, prused, prfound)
end


function _sol_from_ft(inputs, ft, mk)
    for t in inputs.tasks
        t.finish_time = ft[t.id]
        t.start_time = ft[t.id] - t.duration
    end
    sol = Solution()
    sol.tasks = sort(inputs.tasks, by = t -> (t.start_time, t.id))
    sol.makespan = Float64(mk)
    return sol
end
# FBI on a decoded order; returns (makespan, start-time snapshot) of the kept schedule
function _fbi_ft!(s, ft, mk, n)
    st0 = zeros(Int, n)
    for t in s.inputs.tasks
        st0[t.id] = ft[t.id] - t.duration
    end
    sol = _sol_from_ft(s.inputs, ft, mk)
    c0 = SCHEDULE_COUNT[]
    sol2 = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
    s.schedules += SCHEDULE_COUNT[] - c0
    if sol2.makespan < mk
        return Int(sol2.makespan), _snap(sol2, n)
    end
    return Int(mk), st0
end
function _improve!(s, gbest, e, st, mk)
    _offer!(e, st, mk)
    if mk < gbest[]
        gbest[] = mk
        push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        return 1
    end
    return 0
end
function _relink_fbi!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, norig::Int, target::Int)
    idx = sortperm(e.mk)
    length(idx) < 2 && return (0, 0)
    guide = _order(e.st[idx[1]])
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    c_start = s.schedules; found = 0
    for o in idx[2:min(end, norig + 1)]
        order = _order(e.st[o])
        bestmk = typemax(Int); bestft = copy(ft)
        for pos in 1:length(order)
            (s.schedules - c_start >= cap || s.schedules >= target) && break
            order[pos] == guide[pos] && continue
            src = findfirst(==(guide[pos]), order)
            order[pos], order[src] = order[src], order[pos]
            repairPrecedence!(order, s.inputs)
            c0 = SCHEDULE_COUNT[]
            mk = decodeOrder!(s.inputs, s.availability_units, order, ft)
            s.schedules += SCHEDULE_COUNT[] - c0
            if mk < bestmk
                bestmk = mk; bestft .= ft
            end
        end
        bestmk == typemax(Int) && continue
        @assert _order_ok(order, s.inputs)
        s.schedules + 2 > target && break
        mk2, st2 = _fbi_ft!(s, bestft, bestmk, n)
        found += _improve!(s, gbest, e, st2, mk2)
        (s.schedules - c_start >= cap || s.schedules >= target) && break
    end
    return (s.schedules - c_start, found)
end
function _order_ok(order, inputs)
    where = zeros(Int, length(order))
    for (p, id) in enumerate(order)
        where[id] = p
    end
    for t in inputs.tasks, (pid, _) in t.predecessors
        where[pid] < where[t.id] || return false
    end
    return true
end
function _window(order, pos, inputs, where)
    t = inputs.tasks[order[pos]]
    lo = 1; hi = length(order)
    for (pid, _) in t.predecessors
        lo = max(lo, where[pid] + 1)
    end
    for (sid, _) in t.successors
        hi = min(hi, where[sid] - 1)
    end
    return lo, hi
end
function _local_search!(s, e::Elite, n::Int, gbest::Ref{Int}, cap::Int, target::Int)
    idx = sortperm(e.mk)
    cur = _order(e.st[idx[1]]); curmk = e.mk[idx[1]]
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    c_start = s.schedules; found = 0
    L = length(cur)
    @assert decodeOrder!(s.inputs, s.availability_units, cur, ft) <= curmk   # known answer, not counted
    while s.schedules - c_start < cap && s.schedules + 3 <= target
        where = zeros(Int, L)
        for (p, id) in enumerate(cur)
            where[id] = p
        end
        pos = rand(s.rng, 2:L-1)
        lo, hi = _window(cur, pos, s.inputs, where)
        lo == hi && continue
        newpos = rand(s.rng, lo:hi)
        newpos == pos && continue
        cand = copy(cur)
        id = cand[pos]; deleteat!(cand, pos); insert!(cand, newpos, id)
        c0 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c0
        mk <= curmk || continue
        @assert _order_ok(cand, s.inputs)
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        if mk2 <= curmk
            cur = _order(st2); curmk = mk2
            found += _improve!(s, gbest, e, st2, mk2)
        end
    end
    return (s.schedules - c_start, found)
end
function ado_prs2_run!(s, target::Int, mode::Int, e::Elite, n::Int, S::Int, cap::Int, norig::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    runbest = gbest[]
    since = 0; tsince = 0; episodes = 0; used = 0; found = 0
    pols = s.sortPolicies
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, pols)
        sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
        st0 = _snap(sol, n); mk0 = Int(sol.makespan)
        sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        tsince += spent
        s.iter += 1
        mk = Int(sol.makespan)
        _offer!(e, mk < mk0 ? _snap(sol, n) : st0, mk)
        if mk < runbest
            runbest = mk; s.bestBeta = beta; s.bestPolicy = pol
        end
        if mk < gbest[]
            gbest[] = mk; since = 0
            s.bestSolution = sol
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        # mode 5: local search every S schedules of tries, whatever happens (attribution control)
        if (mode == 5 ? tsince >= S : since >= S) && s.schedules < target
            episodes += 1
            u, f = mode == 3 ? _relink_fbi!(s, e, n, gbest, cap, norig, target) :
                               _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            runbest = min(runbest, gbest[])
            since = 0; tsince = 0
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found)
end


mutable struct RevSide
    inputs; pols; at; au
end
# start times in ORIGINAL labels of a schedule built on the reversed instance
function _flip(sol, n)
    C = Int(sol.makespan); v = zeros(Int, n)
    for t in sol.tasks
        v[n + 1 - t.id] = C - (t.start_time + t.duration)
    end
    return v
end
function ado_bidir_start(name::String, revname::String, seed::Int, tieseed::Int, both::Bool)
    mk = Int[]; snaps = Vector{Vector{Int}}(); n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing
    cd(ADO_SOLVER_ROOT) do
        for j in 1:13
            inp = readInputs(name); n = maximum(t.id for t in inp.tasks)
            r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
            s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
            push!(mk, Int(s.bestSolution.makespan)); push!(snaps, _snap(s.bestSolution, n))
        end
        if both
            for j in 1:13
                inp = readInputs(revname)
                r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
                s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(snaps, _flip(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        pols, _ = buildPolicies(inp, Xoshiro(seed), "B2")
        side = RevSide(inp, pols, Dict(x.id => zeros(x.units) for x in inp.resources),
                       zeros(Int64, inp.nResources, inp.timeHorizon + 1))
    end
    for k in eachindex(mk)
        _offer!(e, snaps[k], mk[k])
    end
    m = minimum(mk)
    winners = findall(==(m), mk)
    w = winners[rand(Xoshiro(tieseed), 1:length(winners))]
    ridx = (w - 1) % 13 + 1; back = w > 13
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[ridx])
    end
    if back
        s.bestSolution = Solution(); s.bestSolution.makespan = Float64(m)
    end
    @assert Int(s.bestSolution.makespan) == m
    s.schedules = both ? 78 : 39
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back
end
# pback: probability that a try is built backward (0 and 1 consume no extra random number)
function ado_bidir_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, pback::Float64)
    gbest = Int(s.bestSolution.makespan)
    pols = s.sortPolicies
    nback = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        back = pback >= 1.0 ? true : pback <= 0.0 ? false : rand(s.rng, Float64) < pback
        if back
            nback += 1
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        s.schedules += SCHEDULE_COUNT[] - c0
        s.iter += 1
        if mk < gbest
            gbest = mk; ridx = idx
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest
    return (s.iter, nback)
end
ado_best(e) = (i = argmin(e.mk); (e.mk[i], e.st[i]))


function ado_bidir_ls_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, S::Int, cap::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    since = 0; episodes = 0; used = 0; found = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        if back
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        s.iter += 1
        if mk < gbest[]
            gbest[] = mk; ridx = idx; since = 0
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        if since >= S && s.schedules < target
            episodes += 1
            u, f = _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            since = 0
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found)
end


# dirs: 1 = forward rules only, 3 = forward + backward. Ties: forward first.
function ado_dir_start(name::String, revname::String, seed::Int, tieseed::Int, dirs::Int)
    mk = Int[]; snaps = Vector{Vector{Int}}(); isback = Bool[]; rule = Int[]; n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing
    cd(ADO_SOLVER_ROOT) do
        for (d, fname) in ((1, name), (2, revname))
            dirs & d == 0 && continue
            for j in 1:13
                inp = readInputs(fname); n = maximum(t.id for t in inp.tasks)
                r = Xoshiro(seed); pols, _ = buildPolicies(inp, r, "B2")
                s = startRun(inp, pols, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(rule, j); push!(isback, d == 2)
                push!(snaps, d == 2 ? _flip(s.bestSolution, n) : _snap(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        pols, _ = buildPolicies(inp, Xoshiro(seed), "B2")
        side = RevSide(inp, pols, Dict(x.id => zeros(x.units) for x in inp.resources),
                       zeros(Int64, inp.nResources, inp.timeHorizon + 1))
    end
    for k in eachindex(mk)
        _offer!(e, snaps[k], mk[k])
    end
    m = minimum(mk)
    wf = [k for k in eachindex(mk) if mk[k] == m && !isback[k]]
    wb = [k for k in eachindex(mk) if mk[k] == m && isback[k]]
    pool = isempty(wf) ? wb : wf
    w = pool[rand(Xoshiro(tieseed), 1:length(pool))]
    tie = !isempty(wf) && !isempty(wb)
    ridx = rule[w]; back = isback[w]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols, _ = buildPolicies(inputs, rng, "B2")
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[ridx])
    end
    if back
        s.bestSolution = Solution(); s.bestSolution.makespan = Float64(m)
    end
    @assert Int(s.bestSolution.makespan) == m
    s.schedules = 39 * count_ones(dirs)
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back, tie
end


function ado_pool(inp, rng, kind::String)
    pols, _ = buildPolicies(inp, rng, "B2")
    kind == "k13" && return pols
    kind == "lft" && return [pols[9]]
    kind == "hwru" && return [pols[13]]
    hyb = makeSingleHybridPolicy(inp, rng)
    kind == "hyb" && return hyb
    kind == "k14" && return vcat(pols, hyb)
    error("pool $kind")
end
function ado_gen_start(name::String, revname::String, seed::Int, tieseed::Int, dirs::Int, kind::String)
    mk = Int[]; snaps = Vector{Vector{Int}}(); isback = Bool[]; rule = Int[]; n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing; L = 0
    c0 = SCHEDULE_COUNT[]
    cd(ADO_SOLVER_ROOT) do
        for (d, fname) in ((1, name), (2, revname))
            dirs & d == 0 && continue
            L = length(ado_pool(readInputs(fname), Xoshiro(seed), kind))
            for j in 1:L
                inp = readInputs(fname); n = maximum(t.id for t in inp.tasks)
                r = Xoshiro(seed); pool = ado_pool(inp, r, kind)
                s = startRun(inp, pool, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pool[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(rule, j); push!(isback, d == 2)
                push!(snaps, d == 2 ? _flip(s.bestSolution, n) : _snap(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        side = RevSide(inp, ado_pool(inp, Xoshiro(seed), kind), Dict(x.id => zeros(x.units) for x in inp.resources),
                       zeros(Int64, inp.nResources, inp.timeHorizon + 1))
    end
    init_spent = SCHEDULE_COUNT[] - c0
    @assert init_spent == 3 * L * count_ones(dirs)
    for k in eachindex(mk)
        _offer!(e, snaps[k], mk[k])
    end
    m = minimum(mk)
    wf = [k for k in eachindex(mk) if mk[k] == m && !isback[k]]
    wb = [k for k in eachindex(mk) if mk[k] == m && isback[k]]
    pool = isempty(wf) ? wb : wf
    w = pool[rand(Xoshiro(tieseed), 1:length(pool))]
    tie = !isempty(wf) && !isempty(wb)
    ridx = rule[w]; back = isback[w]
    s = cd(ADO_SOLVER_ROOT) do
        inputs = readInputs(name)
        rng = Xoshiro(seed)
        pols = ado_pool(inputs, rng, kind)
        startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pols[ridx])
    end
    if back
        s.bestSolution = Solution(); s.bestSolution.makespan = Float64(m)
    end
    @assert Int(s.bestSolution.makespan) == m
    s.schedules = init_spent
    s.traj[1] = (iter = 0, schedules = s.schedules, makespan = m, time = 0.0)
    return s, side, e, n, ridx, back, tie
end
function _ado_beta(rng, bmode::Int, bb::Float64, h::Float64)
    bmode == 0 && return rand(rng, Float64)
    bmode == 1 && return rand(rng, TriangularDist(0.0, 1.0, bb))
    c = bmode == 2 ? bb : 0.5
    lo = max(0.0, c - h); hi = min(1.0, c + h)
    return rand(rng, TriangularDist(lo, hi, clamp(c, lo, hi)))
end
# returns (tries, tries_backward, episodes, ls_schedules, ls_records, deep_tries, deep_extra_schedules,
#          deep_records, beta_records, max_overshoot)
function ado_gen_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, ls::Bool, S::Int, cap::Int,
                      bmode::Int, h::Float64, fbimode::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    bb = 0.5
    since = 0; sincet = 0; episodes = 0; used = 0; found = 0
    deep = 0; deepx = 0; deeprec = 0; brec = 0; nb = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = _ado_beta(s.rng, bmode, bb, h)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        isdeep = fbimode < 0 || (fbimode > 0 && sincet >= fbimode)
        passes = isdeep ? typemax(Int64) : 1
        if back
            nb += 1
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = passes)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = passes)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        spent = SCHEDULE_COUNT[] - c0
        if isdeep
            deep += 1; deepx += spent - 3
        end
        s.schedules += spent
        s.iter += 1
        if mk < gbest[]
            gbest[] = mk; ridx = idx; since = 0; sincet = 0
            if bmode in (1, 2)
                bb = beta; brec += 1
            end
            deeprec += isdeep
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
            sincet = (isdeep && fbimode > 0) ? 0 : sincet + 1
        end
        if ls && since >= S && s.schedules < target
            episodes += 1
            u, f = _local_search!(s, e, n, gbest, cap, target)
            used += u; found += f
            since = 0
            f > 0 && (sincet = 0)
        end
    end
    @assert minimum(e.mk) == gbest[]
    return (s.iter, nb, episodes, used, found, deep, deepx, deeprec, brec, s.schedules - target)
end


function _ils_ls!(s, e, n, gbest, cur, curmk, L::Int, target::Int, ft, rng)
    Lc = length(cur); fails = 0; found = 0; idle = 0
    while fails < L && s.schedules + 3 <= target && idle < 50 * L
        where = zeros(Int, Lc)
        for (p, id) in enumerate(cur)
            where[id] = p
        end
        pos = rand(rng, 2:Lc-1)
        lo, hi = _window(cur, pos, s.inputs, where)
        newpos = lo < hi ? rand(rng, lo:hi) : pos
        if newpos == pos
            idle += 1; continue
        end
        idle = 0
        cand = copy(cur)
        id = cand[pos]; deleteat!(cand, pos); insert!(cand, newpos, id)
        c0 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c0
        if mk > curmk
            fails += 1; continue
        end
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        if mk2 <= curmk
            better = mk2 < curmk
            cur = _order(st2); curmk = mk2
            found += _improve!(s, gbest, e, st2, mk2)
            fails = better ? 0 : fails + 1
        else
            fails += 1
        end
    end
    return cur, curmk, found
end
function _ils_perturb(s, base, pert::Int, k::Int, rng, pol)
    Lc = length(base)
    if pert == 4                                   # rm: k random feasible insertion moves
        cur = copy(base); done = 0; tries = 0
        while done < k && tries < 50 * k
            tries += 1
            where = zeros(Int, Lc)
            for (p, id) in enumerate(cur)
                where[id] = p
            end
            pos = rand(rng, 2:Lc-1)
            lo, hi = _window(cur, pos, s.inputs, where)
            lo < hi || continue
            newpos = rand(rng, lo:hi)
            newpos == pos && continue
            id = cur[pos]; deleteat!(cur, pos); insert!(cur, newpos, id); done += 1
        end
        return cur
    end
    kk = min(k, Lc - 2)
    p0 = rand(rng, 2:(Lc - kk))                   # window base[p0 : p0+kk-1], dummies stay put
    seg = base[p0:p0+kk-1]
    out = base[1:p0-1]
    placed = falses(Lc)
    for id in out
        placed[id] = true
    end
    rest = copy(seg)
    beta = rand(rng, Float64)
    while !isempty(rest)
        el = [id for id in rest if all(placed[pid] for (pid, _) in s.inputs.tasks[id].predecessors)]
        @assert !isempty(el)
        if pert == 2
            pick = el[rand(rng, 1:length(el))]
        else
            sort!(el, by = id -> (hybridScore(pol, s.inputs.tasks[id]), id))
            pick = pert == 3 ? el[1] : el[getRandomPosition(beta, length(el), rng)]
        end
        push!(out, pick); placed[pick] = true
        deleteat!(rest, findfirst(==(pick), rest))
    end
    append!(out, base[p0+kk:end])
    return out
end
function _ils_suffix(s, base, kk::Int, rng, pol, uni::Bool)
    Lc = length(base); kk = clamp(kk, 2, Lc - 2)
    p0 = Lc - kk                                    # rebuild base[p0 : Lc-1], the last kk real activities
    out = base[1:p0-1]
    placed = falses(Lc)
    for id in out
        placed[id] = true
    end
    rest = base[p0:Lc-1]
    beta = rand(rng, Float64)
    while !isempty(rest)
        el = [id for id in rest if all(placed[pid] for (pid, _) in s.inputs.tasks[id].predecessors)]
        @assert !isempty(el)
        if uni
            pick = el[rand(rng, 1:length(el))]
        else
            sort!(el, by = id -> (hybridScore(pol, s.inputs.tasks[id]), id))
            pick = el[getRandomPosition(beta, length(el), rng)]
        end
        push!(out, pick); placed[pick] = true
        deleteat!(rest, findfirst(==(pick), rest))
    end
    push!(out, base[Lc])
    return out
end
function _ils_ax!(s, e, n, gbest, target::Int, L::Int, npool::Int, uni::Bool, rng, pol)
    ft = Vector{Int64}(undef, s.inputs.nTasks)
    nreal = length(e.st[1]) - 2
    ord = sortperm(e.mk)[1:min(npool, length(e.mk))]
    bases = [_order(e.st[i]) for i in ord]; bmk = [e.mk[i] for i in ord]
    demon = zeros(Int, length(ord)); p = fill(0.1, length(ord))
    its = 0; acc = 0; rec = 0; j = 0
    if L > 0
        for q in eachindex(bases)
            bases[q], bmk[q], f = _ils_ls!(s, e, n, gbest, bases[q], bmk[q], L, target, ft, rng); rec += f
        end
    end
    while s.schedules + 3 <= target
        its += 1; j = j % length(bases) + 1
        cand = _ils_suffix(s, bases[j], round(Int, p[j] * nreal), rng, pol, uni)
        @assert _order_ok(cand, s.inputs)
        c1 = SCHEDULE_COUNT[]
        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
        s.schedules += SCHEDULE_COUNT[] - c1
        s.schedules + 2 <= target || break
        mk2, st2 = _fbi_ft!(s, ft, mk, n)
        rec += _improve!(s, gbest, e, st2, mk2)
        cur, curmk = _order(st2), mk2
        if L > 0
            cur, curmk, f = _ils_ls!(s, e, n, gbest, cur, curmk, L, target, ft, rng); rec += f
        end
        d = curmk - bmk[j]
        if d < 0
            bases[j], bmk[j] = cur, curmk; demon[j] = -d; p[j] = 0.0; acc += 1
        elseif d < demon[j]
            bases[j], bmk[j] = cur, curmk; demon[j] = 0; p[j] = 0.0; acc += 1
        end
        p[j] = min(p[j] + 0.1, 0.5)
    end
    return its, acc, rec
end
function ado_ils_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, S::Int, cap::Int,
                      pert::Int, k::Int, L::Int, ilsseed::Int, npool::Int)
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    since = 0; episodes = 0; used = 0; found = 0
    its = 0; acc = 0; ilsrec = 0; ils_start = -1; ils_used = 0
    while s.schedules < target
        c0 = SCHEDULE_COUNT[]
        beta = rand(s.rng, Float64)
        pol = rand(s.rng, Float64) < s.policyKeepProb ? pols[ridx] : rand(s.rng, pols)
        idx = findfirst(p -> p === pol, pols)
        if back
            sol = MPJ_Heuristic(side.inputs, side.at, beta, side.pols[idx], true, s.rng)
            st0 = _flip(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, side.inputs, side.au; maxPasses = 1)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _flip(sol, n) : st0
        else
            sol = MPJ_Heuristic(s.inputs, s.availability_times, beta, pol, true, s.rng)
            st0 = _snap(sol, n); mk0 = Int(sol.makespan)
            sol = localSearch(sol, s.inputs, s.availability_units; maxPasses = 1)
            mk = Int(sol.makespan)
            st = mk < mk0 ? _snap(sol, n) : st0
        end
        spent = SCHEDULE_COUNT[] - c0
        s.schedules += spent
        s.iter += 1
        if mk < gbest[]
            gbest[] = mk; ridx = idx; since = 0
            _offer!(e, st, mk)
            push!(s.traj, (iter = s.iter, schedules = s.schedules, makespan = Float64(mk), time = 0.0))
        else
            since += spent
        end
        if since >= S && s.schedules < target
            if pert == 0
                episodes += 1
                u, f = _local_search!(s, e, n, gbest, cap, target)
                used += u; found += f
                since = 0
            else
                # ILS from here to the end of the budget, own random stream
                rng = Xoshiro(ilsseed); pol1 = pols[1]
                ils_start = s.schedules
                if pert >= 6
                    its, acc, ilsrec = _ils_ax!(s, e, n, gbest, target, L, npool, pert == 7, rng, pol1)
                    ils_used = s.schedules - ils_start
                    break
                end
                ft = Vector{Int64}(undef, s.inputs.nTasks)
                i0 = sortperm(e.mk)[1]
                base = _order(e.st[i0]); basemk = e.mk[i0]
                base, basemk, f = _ils_ls!(s, e, n, gbest, base, basemk, L, target, ft, rng); ilsrec += f
                while s.schedules + 3 <= target
                    its += 1
                    if pert == 5
                        cur, curmk = base, basemk
                    else
                        cand = _ils_perturb(s, base, pert, k, rng, pol1)
                        @assert _order_ok(cand, s.inputs)
                        c1 = SCHEDULE_COUNT[]
                        mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
                        s.schedules += SCHEDULE_COUNT[] - c1
                        s.schedules + 2 <= target || break
                        mk2, st2 = _fbi_ft!(s, ft, mk, n)
                        ilsrec += _improve!(s, gbest, e, st2, mk2)
                        cur, curmk = _order(st2), mk2
                    end
                    cur, curmk, f = _ils_ls!(s, e, n, gbest, cur, curmk, L, target, ft, rng); ilsrec += f
                    if curmk <= basemk
                        acc += 1; base, basemk = cur, curmk
                    end
                end
                ils_used = s.schedules - ils_start
                break
            end
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found, its, acc, ilsrec, ils_start, ils_used, s.schedules - target)
end

