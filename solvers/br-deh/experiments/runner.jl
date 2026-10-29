# runner.jl — replication driver for the j30 experiments
#
# Mirrors detExecution() from src/heuristic.jl exactly (same RNG call order)
# but records the improvement trajectory with a dual counter (outer
# iterations AND cumulative SGS schedules), so one 50k-iteration run can be
# read at any checkpoint on either axis.
#
# Configs:
#   A  — as-coded:      FBI loops until no improvement, policy keep-prob 0.5
#   B1 — as-described:  FBI single pass (paper p.11), policy keep-prob 0.8
#                       (paper sec. 5: 20% chance of random policy)
#
# Usage (from repo root, or any cwd — it fixes cwd itself):
#   julia --project=<env> experiments/runner.jl <instance_list> <out_csv> <maxIter> <A|B1>
# or drive runExperiment() from Python via juliacall.

cd(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "heuristic.jl"))
include(joinpath(@__DIR__, "klein_policies.jl"))
include(joinpath(@__DIR__, "pathrelinking.jl"))
include(joinpath(@__DIR__, "process.jl"))

# Same 32 seeds hardcoded in main() (src/heuristic.jl) — kept identical.
const SEEDS = [
    640942923, 697932107, 429007502, 435314368, 425074063, 529343854, 864294084, 85138537,
    310779084, 975978038, 157807440, 293871969, 530462484, 612032821, 60621784, 128299997,
    591873491, 245899912, 620347607, 660916590, 748384338, 907673553, 941867327, 898533368,
    548390311, 512855406, 990831284, 114651404, 193823050, 551855515, 443895884, 794990778,
]

function makeSortPolicies(inputs, rng)
    # Same list, same order as main() in src/heuristic.jl
    return [
        startTimeAndRandomResources(inputs, rng),
        startTimeAndLowUnitResources(inputs, rng),
        startTimeAndHighUnitResources(inputs, rng),
        startTimeAndLowPercentageResources(inputs, rng),
        startTimeAndHighPercentageResources(inputs, rng),
        startTimeAndLowTotalPercentageResources(inputs, rng),
        startTimeAndHighTotalPercentageResources(inputs, rng),
    ]
end

# Mirror of detExecution() with trajectory recording — now a thin wrapper
# over the resumable API in process.jl (startRun/advanceIters!), so both
# paths are the same code and the RNG call sequence stays identical.
function detExecutionTraj(inputs, maxIter, sortPolicies, rng;
                          fbiMaxPasses = typemax(Int64), policyKeepProb = 0.5,
                          initPolicy = nothing, pathRelink = false)
    SCHEDULE_COUNT[] = 0   # legacy behavior: per-run counter starts at zero
    s = startRun(inputs, sortPolicies, rng;
                 fbiMaxPasses = fbiMaxPasses, policyKeepProb = policyKeepProb,
                 initPolicy = initPolicy)
    advanceIters!(s, maxIter)

    # b3: path relinking over the elite pool (Algorithm 1, line 21) — runs
    # after the iteration loop; its schedules land in the counters
    bestMk = s.bestSolution.makespan
    if pathRelink
        tA = time()
        bestMk = pathRelinkingOrders!(s.inputs, s.availability_units, s.elites, bestMk,
            mk -> begin
                s.schedules = SCHEDULE_COUNT[]
                push!(s.traj, (iter = maxIter, schedules = s.schedules,
                               makespan = mk, time = s.elapsed + (time() - tA)))
            end)
        s.schedules = SCHEDULE_COUNT[]
        s.elapsed += time() - tA
    end

    # Terminal record: totals at end of run (same makespan as last improvement)
    push!(s.traj, (iter = maxIter, schedules = s.schedules,
                   makespan = bestMk, time = s.elapsed))

    return s.bestSolution, s.traj
end

function configParams(config)
    config == "A"  && return (fbiMaxPasses = typemax(Int64), policyKeepProb = 0.5)
    config == "B1" && return (fbiMaxPasses = 1,              policyKeepProb = 0.8)
    config == "B2" && return (fbiMaxPasses = 1,              policyKeepProb = 0.8)
    config == "B3" && return (fbiMaxPasses = 1,              policyKeepProb = 0.8, pathRelink = true)
    error("Unknown config: $config (expected A, B1, B2 or B3)")
end

usesKlein(config) = config == "B2" || config == "B3"

function buildPolicies(inputs, rng, config)
    if usesKlein(config)
        pols = makeKleinPolicies(inputs, rng)
        return pols, pols[1]   # initial policy: first of the list (documented choice)
    else
        return makeSortPolicies(inputs, rng), nothing
    end
end

function runExperiment(instanceListFile, outCsv, maxIter, config)
    params = configParams(config)
    instances = filter(!isempty, strip.(readlines(instanceListFile)))

    # JIT warm-up on the first instance so its timings are not inflated
    let inputs = readInputs(instances[1]), rng = Xoshiro(SEEDS[1])
        pols, initPol = buildPolicies(inputs, rng, config)
        detExecutionTraj(inputs, 5, pols, rng; initPolicy = initPol, params...)
    end

    open(outCsv, "w") do io
        println(io, "instance,seed_idx,iter,schedules,makespan,time_s")
        for (n, instanceName) in enumerate(instances)
            for (si, seed) in enumerate(SEEDS)
                # Fresh inputs per (instance, seed): Task objects are mutated
                inputs = readInputs(instanceName)
                rng = Xoshiro(seed)
                sortPolicies, initPol = buildPolicies(inputs, rng, config)
                _, traj = detExecutionTraj(inputs, maxIter, sortPolicies, rng; initPolicy = initPol, params...)
                for p in traj
                    println(io, "$instanceName,$si,$(p.iter),$(p.schedules),$(p.makespan),$(round(p.time, digits = 4))")
                end
            end
            flush(io)
            println(stderr, "[$config] $n/$(length(instances)) $instanceName done")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 4 || error("Usage: julia runner.jl <instance_list> <out_csv> <maxIter> <A|B1>")
    runExperiment(ARGS[1], ARGS[2], parse(Int64, ARGS[3]), ARGS[4])
end
