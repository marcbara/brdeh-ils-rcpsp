# process.jl — budget-driven, resumable execution of the BR-DEH search.
#
# A RunState holds the full search state (incumbent beta/policy, best
# solution, elites, RNG, caches, counters) so a run can be advanced in
# slices and read at schedule checkpoints, with results identical to one
# uninterrupted long run of the same total length:
#
#   s = startRun(inputs, pols, rng; fbiMaxPasses = 1, policyKeepProb = 0.8)
#   advanceRun!(s, 2500)   # run until >= 2500 schedules spent (iteration-
#   advanceRun!(s, 2500)   # aligned: the last iteration may overshoot a few)
#   s.traj                 # improvement rows (iter, schedules, makespan, time)
#
# Semantics:
#   - Budgets are in generated SGS schedules. advanceRun! stops at the end
#     of the first iteration that reaches the target, reports the real
#     spend via s.schedules (the caller can discount any excess from its
#     next slice). For results "at budget B", count only improvement rows
#     with schedules <= B.
#   - Each RunState needs its own freshly-read `inputs` (Task objects are
#     mutated during search) and its own RNG.
#   - Time (s.elapsed, and the time field of rows) accumulates solver-only
#     wall time across advances; whatever happens between advances does not
#     count. As in detExecutionTraj, the initial solution's time is excluded.
#   - Path relinking (config B3) is not supported here: it runs after the
#     iteration loop in the original algorithm and has no slice-consistent
#     semantics.
#
# detExecutionTraj (runner.jl) is a thin wrapper over this API, so both
# paths are the same code.

mutable struct RunState
    inputs
    sortPolicies
    rng
    fbiMaxPasses :: Int64
    policyKeepProb :: Float64
    availability_times
    availability_units
    bestBeta :: Float64
    bestPolicy
    bestSolution
    elites
    traj :: Vector{NamedTuple{(:iter, :schedules, :makespan, :time),
                              Tuple{Int64, Int64, Float64, Float64}}}
    iter :: Int64
    schedules :: Int64
    elapsed :: Float64
end

function startRun(inputs, sortPolicies, rng;
                  fbiMaxPasses = typemax(Int64), policyKeepProb = 0.5,
                  initPolicy = nothing)
    bestBeta = 0.5
    bestPolicy = initPolicy === nothing ? startTimeAndHighUnitResources(inputs, rng) : initPolicy

    availability_times = Dict(r.id => zeros(r.units) for r in inputs.resources)
    availability_units = zeros(Int64, inputs.nResources, inputs.timeHorizon + 1)

    traj = Vector{NamedTuple{(:iter, :schedules, :makespan, :time),
                             Tuple{Int64, Int64, Float64, Float64}}}()

    c0 = SCHEDULE_COUNT[]
    beta = bestBeta
    sortPolicy = bestPolicy
    initSolution = MPJ_Heuristic(inputs, availability_times, beta, sortPolicy, false, rng)
    initSolution = localSearch(initSolution, inputs, availability_units; maxPasses = fbiMaxPasses)
    bestSolution = initSolution
    spent = SCHEDULE_COUNT[] - c0

    push!(traj, (iter = 0, schedules = spent,
                 makespan = bestSolution.makespan, time = 0.0))

    # Elite pool as id permutations (Solution objects alias mutable Tasks)
    elites = [(order = [t.id for t in bestSolution.tasks], makespan = bestSolution.makespan)]

    return RunState(inputs, sortPolicies, rng, fbiMaxPasses, policyKeepProb,
                    availability_times, availability_units,
                    bestBeta, bestPolicy, bestSolution, elites, traj,
                    0, spent, 0.0)
end

# One search iteration — exact mirror of the loop body of detExecution()
# (same RNG draw order: TriangularDist beta, policy coin, construction).
@inline function _iterate!(s::RunState)
    c0 = SCHEDULE_COUNT[]
    beta = rand(s.rng, TriangularDist(0.0, 1.0, s.bestBeta))
    sortPolicy = rand(s.rng, Float64) < s.policyKeepProb ? s.bestPolicy : rand(s.rng, s.sortPolicies)
    solution = MPJ_Heuristic(s.inputs, s.availability_times, beta, sortPolicy, true, s.rng)
    solution = localSearch(solution, s.inputs, s.availability_units; maxPasses = s.fbiMaxPasses)
    s.schedules += SCHEDULE_COUNT[] - c0
    improved = solution.makespan < s.bestSolution.makespan
    if improved
        s.bestBeta = beta
        s.bestPolicy = sortPolicy
        s.bestSolution = solution
    end
    s.iter += 1
    return improved
end

@inline function _record!(s::RunState, tAdv::Float64)
    push!(s.traj, (iter = s.iter, schedules = s.schedules,
                   makespan = s.bestSolution.makespan, time = s.elapsed + (time() - tAdv)))
    push!(s.elites, (order = [t.id for t in s.bestSolution.tasks],
                     makespan = s.bestSolution.makespan))
end

"""Advance until at least `budget` MORE schedules are spent (iteration-
aligned). Returns the schedules actually spent in this slice."""
function advanceRun!(s::RunState, budget::Integer)
    target = s.schedules + budget
    tAdv = time()
    while s.schedules < target
        _iterate!(s) && _record!(s, tAdv)
    end
    s.elapsed += time() - tAdv
    return s.schedules - (target - budget)
end

"""Advance exactly `n` more iterations (the legacy, iteration-driven mode)."""
function advanceIters!(s::RunState, n::Integer)
    tAdv = time()
    for _ in 1:n
        _iterate!(s) && _record!(s, tAdv)
    end
    s.elapsed += time() - tAdv
    return s.schedules
end
