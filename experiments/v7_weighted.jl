# v7_weighted.jl — V7: the tie-break rule inside the ILS.
#
# V6 spends ~98 % of its budget in the ILS, and every perturbation rebuilds a
# window ordered by ONE fixed composite rule (LFT/HWRU 50/50). V7 replaces that
# rule by a weighted family and varies or adapts the weight:
#
#   score_w(t) = w * LFTnorm(t) + (1 - w) * (1 - HWRUnorm(t))     (lower first)
#
# with w in {0, 0.1, ..., 1}. w = 0.5 is exactly the V5/V6 rule (HybridPolicy
# mode 1 in escoto_hybrid_policies_90f8ab1.jl, same instance-wide normalization).
#
# Arms (see v7_driver.jl):
#   V6   original V6 (ado_gen_start "hyb" + ado_ils_run!), untouched
#   V7a  start and tries over the 11 weights (66 start schedules, charged);
#        the ILS keeps the weight of the incumbent rule at the switch
#   V7r  control for V7a: same start, but the ILS uses a weight drawn at random
#   V7b  start and tries as V6; every ILS perturbation draws w uniformly
#   V7c  start and tries as V6; every ILS perturbation draws w with
#        probabilities adapted to the weights that improved the base
#
# Requires, included first: harness/solvers/brdeh_bridge.jl,
# experiments/escoto_hybrid_policies_90f8ab1.jl, experiments/extensions.jl.

struct WeightedHybrid
    w :: Float64
    h :: HybridData
end

@inline hybridScore(p::WeightedHybrid, t) =
    p.w * p.h.lftNorm[t.id] + (1.0 - p.w) * (1.0 - p.h.hwruNorm[t.id])

@inline function (p::WeightedHybrid)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return hybridScore(p, a) < hybridScore(p, b)
end

const V7_GRID = [k / 10 for k in 0:10]

function v7_grid(inp)
    d = kleinData(inp)
    caps = Vector{Int64}(undef, inp.nResources)
    for r in inp.resources
        caps[r.id] = r.units
    end
    h = hybridData(inp, d, caps)
    return [WeightedHybrid(w, h) for w in V7_GRID]
end

# Start/tries pools:
#   hyb  the V5/V6 rule alone
#   w11  the 11 weights of the LFT/HWRU family
#   h7   the 7 HybridPolicy modes of escoto_hybrid_policies_90f8ab1.jl
#        (lft_hwru, hrpw_hrd, stfs_hwru, lft_hwru 0.3/0.7 and 0.7/0.3, lft_hwru_hrd, 4-way)
#   p21  the 21 pairwise average blends of 7 criteria (lft, hwru, hrpw*, hrd, stfs, mis, eft)
#   p42  p21 plus the same 21 pairs with max ("or") semantics
#   p55  average blends of every pair of the 11 STATIC Klein criteria (the 13 of
#        klein_policies.jl without the dynamic EFTD and STFD), each min-max
#        normalized over the instance and oriented so that lower is favored

# The 11 static Klein modes: SAD MIS MTS HRPW HRPW* EFT LST LFT STFS HRD HWRU
const V7_STATIC_MODES = Int8[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13]

struct StaticPair
    i :: Int
    j :: Int
    F :: Matrix{Float64}   # F[c, task id] = favored-low normalized value of criterion c
end
@inline hybridScore(p::StaticPair, t) = 0.5 * p.F[p.i, t.id] + 0.5 * p.F[p.j, t.id]
@inline function (p::StaticPair)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return hybridScore(p, a) < hybridScore(p, b)
end

function v7_static_pairs(inp)
    kp = makeKleinPolicies(inp, nothing)
    C = length(V7_STATIC_MODES)
    F = zeros(C, inp.nTasks)
    for (c, m) in enumerate(V7_STATIC_MODES)
        raw = [kleinValue(kp[m], t) for t in inp.tasks]   # static modes ignore start_time
        lo, hi = extrema(raw)
        norm = hi == lo ? zeros(length(raw)) : (raw .- lo) ./ (hi - lo)
        for (k, t) in enumerate(inp.tasks)
            F[c, t.id] = kleinMaximize(Int8(m)) ? 1.0 - norm[k] : norm[k]
        end
    end
    return [StaticPair(i, j, F) for i in 1:C for j in (i+1):C]
end

function v7_pool(inp, rng, kind::String)
    kind == "hyb" && return makeSingleHybridPolicy(inp, rng)
    kind == "w11" && return v7_grid(inp)
    if kind == "h7"
        d = kleinData(inp)
        caps = Vector{Int64}(undef, inp.nResources)
        for r in inp.resources
            caps[r.id] = r.units
        end
        h = hybridData(inp, d, caps)
        return [HybridPolicy(Int8(m), h) for m in 1:7]
    end
    kind == "p21" && return makeAllPairwisePolicies(inp, rng)
    kind == "p42" && return vcat(makeAllPairwisePolicies(inp, rng), makeAllPairwisePoliciesMax(inp, rng))
    kind == "p55" && return v7_static_pairs(inp)
    error("v7 pool $kind")
end

# Copy of ado_gen_start (extensions.jl) with v7_pool instead of ado_pool.
# With kind = "hyb" it is identical to ado_gen_start(..., "hyb").
function v7_gen_start(name::String, revname::String, seed::Int, tieseed::Int, dirs::Int, kind::String)
    mk = Int[]; snaps = Vector{Vector{Int}}(); isback = Bool[]; rule = Int[]; n = 0
    e = Elite(20, Vector{Int}[], Int[], UInt64[])
    side = nothing; L = 0
    c0 = SCHEDULE_COUNT[]
    cd(ADO_SOLVER_ROOT) do
        for (d, fname) in ((1, name), (2, revname))
            dirs & d == 0 && continue
            L = length(v7_pool(readInputs(fname), Xoshiro(seed), kind))
            for j in 1:L
                inp = readInputs(fname); n = maximum(t.id for t in inp.tasks)
                r = Xoshiro(seed); pool = v7_pool(inp, r, kind)
                s = startRun(inp, pool, r; fbiMaxPasses = 1, policyKeepProb = 0.8, initPolicy = pool[j])
                push!(mk, Int(s.bestSolution.makespan)); push!(rule, j); push!(isback, d == 2)
                push!(snaps, d == 2 ? _flip(s.bestSolution, n) : _snap(s.bestSolution, n))
            end
        end
        inp = readInputs(revname)
        side = RevSide(inp, v7_pool(inp, Xoshiro(seed), kind), Dict(x.id => zeros(x.units) for x in inp.resources),
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
        pols = v7_pool(inputs, rng, kind)
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

# Weight picker for one ILS perturbation.
#   ilsmode 0: fixed index `fixed` (V7a: incumbent rule at the switch)
#   ilsmode 1: uniform over the grid (V7b)
#   ilsmode 2: adaptive (V7c): p_j = pmin + (1 - G*pmin) * q_j / sum(q), with
#              q_j <- 0.9 q_j + 0.1 r after each use, r = 1 if the base strictly
#              improved, 0.3 if the candidate was accepted at equal makespan, else 0
#   ilsmode 3: fixed index drawn once at random (V7r)
const V7_PMIN = 0.02

function _v7_pick(rng, ilsmode::Int, fixed::Int, q::Vector{Float64})
    ilsmode == 0 && return fixed
    ilsmode == 3 && return fixed
    G = length(q)
    ilsmode == 1 && return rand(rng, 1:G)
    tot = sum(q); u = rand(rng, Float64); acc = 0.0
    for j in 1:G
        acc += V7_PMIN + (1 - G * V7_PMIN) * q[j] / tot
        u < acc && return j
    end
    return G
end

# Copy of ado_ils_run! (extensions.jl) restricted to the biased window rebuild
# (pert = 1, npool = 1), with the rebuild rule chosen per perturbation.
# Returns the same 10 counters as ado_ils_run! plus (wfix, uses, improvements).
# ilsKind picks the rule set sampled inside the ILS: "w11" (the 11 weights) or any
# v7_pool kind, e.g. "p21" (the 21 criterion pairs) or "h7" (the 7 hybrids).
@inline hybridScore(p::PairwisePolicy, t) = pairScore(p, t)

function v7_ils_run!(s, side, target::Int, e::Elite, n::Int, ridx::Int, back::Bool, S::Int, cap::Int,
                     k::Int, L::Int, ilsseed::Int, ilsmode::Int, gridIsPool::Bool; ilsKind::String = "w11")
    gbest = Ref(Int(s.bestSolution.makespan))
    pols = s.sortPolicies
    since = 0; episodes = 0; used = 0; found = 0
    its = 0; acc = 0; ilsrec = 0; ils_start = -1; ils_used = 0
    ilsKind == "w11" || ilsmode in (1, 2) || error("ilsKind $ilsKind needs a sampling ilsmode (1 or 2)")
    G = ilsKind == "w11" ? length(V7_GRID) : length(v7_pool(s.inputs, nothing, ilsKind))
    uses = zeros(Int, G); impr = zeros(Int, G); wfix = 0
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
            # ILS from here to the end of the budget, own random stream (as V6)
            rng = Xoshiro(ilsseed)
            grid = ilsKind == "w11" ? v7_grid(s.inputs) : v7_pool(s.inputs, nothing, ilsKind)
            q = fill(0.5, G)
            if ilsmode == 0
                # the weight of the incumbent rule; with the "hyb" pool that is w = 0.5
                wfix = gridIsPool ? ridx : findfirst(==(0.5), V7_GRID)
            elseif ilsmode == 3
                wfix = rand(rng, 1:G)
            end
            ils_start = s.schedules
            ft = Vector{Int64}(undef, s.inputs.nTasks)
            i0 = sortperm(e.mk)[1]
            base = _order(e.st[i0]); basemk = e.mk[i0]
            base, basemk, f = _ils_ls!(s, e, n, gbest, base, basemk, L, target, ft, rng); ilsrec += f
            while s.schedules + 3 <= target
                its += 1
                j = _v7_pick(rng, ilsmode, wfix, q)
                cand = _ils_perturb(s, base, 1, k, rng, grid[j])
                @assert _order_ok(cand, s.inputs)
                c1 = SCHEDULE_COUNT[]
                mk = decodeOrder!(s.inputs, s.availability_units, cand, ft)
                s.schedules += SCHEDULE_COUNT[] - c1
                s.schedules + 2 <= target || break
                mk2, st2 = _fbi_ft!(s, ft, mk, n)
                ilsrec += _improve!(s, gbest, e, st2, mk2)
                cur, curmk = _order(st2), mk2
                cur, curmk, f = _ils_ls!(s, e, n, gbest, cur, curmk, L, target, ft, rng); ilsrec += f
                uses[j] += 1
                better = curmk < basemk
                accepted = curmk <= basemk
                better && (impr[j] += 1)
                if ilsmode == 2
                    q[j] = 0.9 * q[j] + 0.1 * (better ? 1.0 : accepted ? 0.3 : 0.0)
                end
                if accepted
                    acc += 1; base, basemk = cur, curmk
                end
            end
            ils_used = s.schedules - ils_start
            break
        end
    end
    @assert s.schedules <= target + 2
    @assert minimum(e.mk) == gbest[]
    return (s.iter, episodes, used, found, its, acc, ilsrec, ils_start, ils_used, s.schedules - target), wfix, uses, impr
end
