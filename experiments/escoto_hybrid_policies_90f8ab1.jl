# hybrid_policies.jl — goal round: NEW tie-break policies (not from Klein
# 1999, not from the paper) that combine two of Klein's 13 criteria into
# one composite score, on the hypothesis that a single criterion may not
# capture enough information but a weighted blend of a time-criticality
# signal (LFT: how much slack this task has before it delays the project)
# and a resource-pressure signal (HWRU: how scarce/contended its resources
# are) might make better tie-break decisions than either alone — untested
# by rounds 1-3/goal (those only ever picked among the EXISTING 13 Klein
# policies via bandits/rules, never created new tie-break criteria).
#
# Reuses KleinData/kleinValue from klein_policies.jl (must be included
# first) so the LFT/HRD/HWRU raw values don't need recomputing.
#
# 3 hybrids, each a weighted min-normalized blend (lower composite score
# wins, like most of Klein's min-policies):
#   - hybrid_lft_hwru:  0.5*normalized(LFT) + 0.5*normalized(-HWRU)
#     (favor tasks close to the critical path AND high resource pressure —
#     schedule the "urgent and contended" tasks first)
#   - hybrid_hrpw_hrd:  0.5*normalized(-HRPW*) + 0.5*normalized(-HRD)
#     (favor tasks with many heavy successors AND high own resource
#     demand — clear the "big blockers" early)
#   - hybrid_stfs_hwru: 0.5*normalized(STFS) + 0.5*normalized(-HWRU)
#     (favor low-slack tasks AND high resource pressure, a variant of the
#     first hybrid using static instead of dynamic float)
#
# Normalization: min-max over the CURRENT candidate list at comparison
# time is not available inside a 2-arg comparator (sort! only ever sees
# pairs, not the whole list) — Underspecification resolved here: normalize
# by the INSTANCE-WIDE min/max of that raw value across all tasks
# (precomputed once in HybridData, like KleinData), not the candidate
# list's own min/max. This is a documented approximation: a task's
# relative rank within the full instance is used as a proxy for its rank
# within the (much smaller) candidate list at any given moment.

struct HybridData
    lftNorm :: Vector{Float64}    # (lft - min) / (max - min), 0=best(min) .. 1=worst
    hwruNorm :: Vector{Float64}   # same, so "-HWRU favored" becomes minimizing hwruNorm's complement
    hrpwStarNorm :: Vector{Float64}
    hrdNorm :: Vector{Float64}
    stfsNorm :: Vector{Float64}
    misNorm :: Vector{Float64}    # goal round v3: MIS (successor count), EFT added to the criterion pool
    eftNorm :: Vector{Float64}
end

@inline function minmaxNorm(values)
    lo, hi = extrema(values)
    span = hi - lo
    span == 0 && return zeros(length(values))
    return [(v - lo) / span for v in values]
end

function hybridData(inputs, d::KleinData, caps::Vector{Int64})
    n = inputs.nTasks
    lft = d.lft
    hwru = zeros(n)
    hrpwStar = zeros(n)
    hrd = zeros(n)
    stfs = d.lst .- d.est
    for task in inputs.tasks
        s = 0.0
        for (r, u) in task.resources
            s += u / caps[r]
        end
        hwru[task.id] = HWRU_OMEGA * length(task.successors) + (1 + HWRU_OMEGA) * s

        hs = task.duration
        for sid in d.fullsucc[task.id]
            hs += inputs.tasks[sid].duration
        end
        hrpwStar[task.id] = hs

        rd = 0
        for (_, u) in task.resources
            rd += u
        end
        hrd[task.id] = task.duration * rd
    end
    mis = Float64[length(t.successors) for t in inputs.tasks]
    eft = d.eft
    return HybridData(minmaxNorm(lft), minmaxNorm(hwru), minmaxNorm(hrpwStar), minmaxNorm(hrd), minmaxNorm(stfs),
                       minmaxNorm(mis), minmaxNorm(eft))
end

struct HybridPolicy
    mode :: Int8   # 1=lft_hwru 2=hrpw_hrd 3=stfs_hwru
                  # goal round v2 (weight sweep + 3-way blend, since the
                  # 50/50 2-way blends beat Klein's 13 singles but the
                  # weighting/arity itself was never varied):
                  # 4=lft_hwru_resourceHeavy (0.3/0.7) 5=lft_hwru_timeHeavy (0.7/0.3)
                  # 6=lft_hwru_hrd_triple (equal 3-way blend)
    h :: HybridData
end

@inline function hybridScore(p::HybridPolicy, t)
    m = p.mode
    # all scores built to be MINIMIZED (lower = more urgent / more favored)
    m == Int8(1) && return 0.5 * p.h.lftNorm[t.id]      + 0.5 * (1.0 - p.h.hwruNorm[t.id])
    m == Int8(2) && return 0.5 * (1.0 - p.h.hrpwStarNorm[t.id]) + 0.5 * (1.0 - p.h.hrdNorm[t.id])
    m == Int8(3) && return 0.5 * p.h.stfsNorm[t.id]       + 0.5 * (1.0 - p.h.hwruNorm[t.id])
    m == Int8(4) && return 0.3 * p.h.lftNorm[t.id]      + 0.7 * (1.0 - p.h.hwruNorm[t.id])
    m == Int8(5) && return 0.7 * p.h.lftNorm[t.id]      + 0.3 * (1.0 - p.h.hwruNorm[t.id])
    m == Int8(6) && return (1.0/3) * p.h.lftNorm[t.id] + (1.0/3) * (1.0 - p.h.hwruNorm[t.id]) + (1.0/3) * (1.0 - p.h.hrdNorm[t.id])
    # goal round v5: 4-way blend of the union of all 3 original winning
    # hybrids' criteria (LFT, HWRU, HRD, STFS) — a single "grand unified"
    # policy instead of 3 separate 2-criterion ones, on the hypothesis
    # that combining the winners' own signal set (not fresh Klein criteria)
    # into one score might beat picking among the 3 separately.
    return 0.25 * p.h.lftNorm[t.id] + 0.25 * (1.0 - p.h.hwruNorm[t.id]) +
           0.25 * (1.0 - p.h.hrdNorm[t.id]) + 0.25 * p.h.stfsNorm[t.id]
end

@inline function (p::HybridPolicy)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return hybridScore(p, a) < hybridScore(p, b)
end

# Goal round v3: exhaustive pairwise screening across ALL 7 criteria (not
# just the 3 the first hand-picked hybrid triad used), to check whether
# the space of 2-criterion blends is actually exhausted or whether the
# 3 original hybrids just got lucky. Each criterion is stored as
# "favored when LOW" (norm) — a `sign` of +1 uses the norm directly
# (favor low raw value, like LFT/STFS), -1 uses (1-norm) (favor high raw
# value, like HWRU/HRPW*/HRD/MIS).
const CRITERIA_NAMES = ["lft", "hwru", "hrpwStar", "hrd", "stfs", "mis", "eft"]
const CRITERIA_SIGN  = [ +1,    -1,     -1,         -1,    +1,     -1,   +1]  # +1 favor-low, -1 favor-high

@inline function critNorm(h::HybridData, idx::Integer, t)
    idx == 1 && return h.lftNorm[t.id]
    idx == 2 && return h.hwruNorm[t.id]
    idx == 3 && return h.hrpwStarNorm[t.id]
    idx == 4 && return h.hrdNorm[t.id]
    idx == 5 && return h.stfsNorm[t.id]
    idx == 6 && return h.misNorm[t.id]
    return h.eftNorm[t.id]
end

struct PairwisePolicy
    i :: Int8
    j :: Int8
    h :: HybridData
    useMax :: Bool   # goal round v4: false=average blend (the 21 already
                      # tested), true=MAX of the two normalized-favored
                      # scores (an "OR" semantics — favor a task that's
                      # extreme on EITHER criterion, not just the two-
                      # criterion average — untested combination logic)
end

@inline function pairScore(p::PairwisePolicy, t)
    sa = CRITERIA_SIGN[p.i] > 0 ? critNorm(p.h, p.i, t) : (1.0 - critNorm(p.h, p.i, t))
    sb = CRITERIA_SIGN[p.j] > 0 ? critNorm(p.h, p.j, t) : (1.0 - critNorm(p.h, p.j, t))
    return p.useMax ? max(sa, sb) : 0.5*sa + 0.5*sb
end

@inline function (p::PairwisePolicy)(a, b)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    return pairScore(p, a) < pairScore(p, b)
end

# All C(7,2)=21 pairwise combinations, as an extra policy pool for
# exploratory screening (not meant to be used alongside the original 3 —
# a dedicated pool so a strategy can pick among all 21 at once).
function makeAllPairwisePolicies(inputs, rng; useMax = false)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)
    pols = PairwisePolicy[]
    for i in 1:7, j in (i+1):7
        push!(pols, PairwisePolicy(Int8(i), Int8(j), h, useMax))
    end
    return pols
end

# goal round v4: same 21 pairs but with MAX ("OR") semantics instead of
# average ("AND") — see PairwisePolicy.useMax.
makeAllPairwisePoliciesMax(inputs, rng) = makeAllPairwisePolicies(inputs, rng; useMax = true)

function makeHybridPolicies(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)
    return [HybridPolicy(Int8(m), h) for m in 1:3]
end

# goal round v5: single 4-way "grand unified" blend (mode 7) as its own
# 1-policy pool — no selection needed, always the same score.
# goal round v7: pick the single best hybrid mode ONCE, based on
# instance-level structural data computed at read time — not learned
# during search (unlike every bandit/rule tried before), and not fixed
# across all instances (unlike makeSingleHybridPolicy). Computes a
# resource-density proxy (mean per-task resource demand as a fraction of
# available capacity) and picks the resource-pressure-oriented hybrid
# (mode 2, hrpw_hrd) for tight instances, the time-criticality-oriented
# one (mode 1, lft_hwru) for loose ones — the hypothesis rounds 1-3/goal
# never tested: maybe no SINGLE mode wins everywhere, and the right
# choice is instance-dependent, decided once instead of learned.
function makeInstanceAdaptivePolicy(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)

    totalDemand = 0.0
    totalCapacitySlots = 0.0
    for task in inputs.tasks
        for (r, u) in task.resources
            totalDemand += u * task.duration
            totalCapacitySlots += caps[r] * task.duration
        end
    end
    density = totalCapacitySlots > 0 ? totalDemand / totalCapacitySlots : 0.0

    # Threshold calibrated against observed densities across j30-j120
    # samples (typically 0.1-0.4, never near the naive 0.5 guess).
    mode = density > 0.2 ? Int8(2) : Int8(1)
    return [HybridPolicy(mode, h)]
end

function makeGrandHybridPolicy(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)
    return [HybridPolicy(Int8(7), h)]
end

# goal round v5b: the single BEST original hybrid (mode 1, lft_hwru) alone,
# no alternation at all — checks whether random_08's 20% "explore the
# other 2 hybrids" draws are actively hurting vs. just always exploiting
# the single strongest one.
function makeSingleHybridPolicy(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)
    return [HybridPolicy(Int8(1), h)]
end

# Goal round v2: all 6 hybrid modes (original 3 + weight-sweep/triple-blend
# variants), so a selection strategy could learn which weighting suits
# which instance rather than being stuck with the original modes 1-3.
function makeHybridPoliciesV2(inputs, rng)
    d = kleinData(inputs)
    caps = Vector{Int64}(undef, inputs.nResources)
    for r in inputs.resources
        caps[r.id] = r.units
    end
    h = hybridData(inputs, d, caps)
    return [HybridPolicy(Int8(m), h) for m in 1:6]
end

# Extended pool: Klein's 13 + these 3 hybrids (16 total), for strategies
# that want the widest policy universe to select/learn over.
function makeExtendedPolicies(inputs, rng)
    return vcat(makeKleinPolicies(inputs, rng), makeHybridPolicies(inputs, rng))
end
