# brdeh_bridge.jl — glue between the standalone artifact harness and the BR-DEH runner.
#
# Defines no search logic: it only calls the solver's resumable API
# (experiments/process.jl: startRun / advanceRun!) and the runner's helpers
# (readInputs, buildPolicies, configParams, SEEDS), so a harness process is
# the same code as a legacy runExperiment() run. The solver version is
# whatever commit the submodule points at.

const ADO_SOLVER_ROOT = abspath(joinpath(@__DIR__, "..", "..", "solvers", "br-deh"))

# runner.jl cd()s into the solver root when included; that would silently
# change the Python process cwd too, so restore it afterwards.
let here = pwd()
    include(joinpath(ADO_SOLVER_ROOT, "experiments", "runner.jl"))
    cd(here)
end

# New search process with fresh inputs (Tasks are mutated) and its own RNG.
# `klein` picks the policy list through the solver's own buildPolicies.
# startRun builds the initial solution: its schedules count, its time does
# not (solver convention).
function ado_start(instanceName::String, seed::Int, klein::Bool,
                   fbiMaxPasses::Int, policyKeepProb::Float64)
    cd(ADO_SOLVER_ROOT) do   # readInputs() reads "inputs/" relative to cwd
        inputs = readInputs(instanceName)
        rng = Xoshiro(seed)
        pols, initPol = buildPolicies(inputs, rng, klein ? "B2" : "A")
        startRun(inputs, pols, rng; fbiMaxPasses = fbiMaxPasses,
                 policyKeepProb = policyKeepProb, initPolicy = initPol)
    end
end

# Advance by >= b schedules (iteration-aligned); b <= 0 is a no-op.
ado_advance!(s, b::Int) = b > 0 ? advanceRun!(s, b) : 0

# (schedules spent, solver seconds, number of trajectory rows)
ado_counters(s) = (s.schedules, s.elapsed, length(s.traj))

# Julia runtime counters for the time-axis diagnostics: cumulative GC time
# and JIT compile time (ns), threads and GC threads of this runtime.
ado_runtime_counters() = (Base.gc_time_ns(), Base.cumulative_compile_time_ns()[1])
ado_runtime_info() = (nthreads = Threads.nthreads(), ngcthreads = Threads.ngcthreads(),
                      julia = string(VERSION))

# Trajectory rows from index `from` (1-based) on, as an N×4 Float64 matrix:
# iter, schedules, makespan, time_s.
function ado_rows(s, from::Int)
    n = length(s.traj) - from + 1
    out = Matrix{Float64}(undef, max(n, 0), 4)
    for k in 1:n
        p = s.traj[from + k - 1]
        out[k, 1] = p.iter; out[k, 2] = p.schedules
        out[k, 3] = p.makespan; out[k, 4] = p.time
    end
    out
end

# Snapshot the incumbent in solver task-id order for independent validation.
function ado_state_starts(s)
    # RunState.bestSolution aliases mutable Tasks, so reconstruct its elite
    # id-order through the solver's own serial SGS before independent checks.
    best = argmin([e.makespan for e in s.elites])
    finish = Vector{Int64}(undef, s.inputs.nTasks)
    avail = zeros(Int64, s.inputs.nResources, s.inputs.timeHorizon + 1)
    mk = Int(decodeOrder!(s.inputs, avail, s.elites[best].order, finish))
    out = [Int(finish[t.id] - t.duration) for t in s.inputs.tasks]
    return mk, out
end

# Validation-only replay of an elite order matching a cap-visible makespan.
function ado_state_starts_at(s, wanted::Int)
    for e in s.elites
        Int(e.makespan) == wanted || continue
        finish = Vector{Int64}(undef, s.inputs.nTasks)
        avail = zeros(Int64, s.inputs.nResources, s.inputs.timeHorizon + 1)
        mk = Int(decodeOrder!(s.inputs, avail, e.order, finish))
        mk == wanted || continue
        return mk, [Int(finish[t.id] - t.duration) for t in s.inputs.tasks]
    end
    error("no replayable elite for makespan $wanted")
end
function ado_elite_starts_at(e, wanted::Int)
    i = findfirst(==(wanted), e.mk)
    i === nothing && error("no elite snapshot for makespan $wanted")
    return e.mk[i], e.st[i]
end

# The solver's own preset table, exported so the Python side can check that
# its named presets still match configParams() of the pinned solver commit.
function ado_config_params(config::String)
    p = configParams(config)
    (fbiMaxPasses = p.fbiMaxPasses, policyKeepProb = p.policyKeepProb,
     pathRelink = get(p, :pathRelink, false), klein = usesKlein(config))
end

# JIT warm-up so the first timed call is not inflated (runner discipline).
function ado_warmup()
    for (klein, fbi, keep) in ((true, 1, 0.8), (false, typemax(Int64), 0.5))
        s = ado_start("j30/j301_1.sm", SEEDS[1], klein, fbi, keep)
        ado_advance!(s, 30); ado_counters(s); ado_rows(s, 1)
    end
    nothing
end
