include(joinpath(@__DIR__, "..", "harness", "solvers", "brdeh_bridge.jl"))
include(joinpath(@__DIR__, "..", "experiments", "escoto_hybrid_policies_90f8ab1.jl"))
include(joinpath(@__DIR__, "..", "experiments", "extensions.jl"))

const B = parse(Int, get(ENV, "BRDEH_SMOKE_BUDGET", "39"))
const SEED = SEEDS[1]
const MODE = isempty(ARGS) ? "all" : ARGS[1]
function trajectory(s)
    [(iter=Int(r.iter), schedules=Int(r.schedules), makespan=Int(r.makespan)) for r in s.traj]
end
function check_elite(instance, e)
    i = argmin(e.mk); inp = cd(() -> readInputs(instance), ADO_SOLVER_ROOT)
    starts = e.st[i]
    for task in inp.tasks, (pid, _) in task.predecessors
        starts[pid] + inp.tasks[pid].duration <= starts[task.id] || return false
    end
    for resource in inp.resources
        for t in 0:Int(e.mk[i])-1
            used = 0
            for task in inp.tasks
                for (rid, units) in task.resources
                    rid == resource.id && starts[task.id] <= t < starts[task.id] + task.duration && (used += units)
                end
            end
            used <= resource.units || return false
        end
    end
    return true
end

if MODE == "v0"
    println("V0")
    for instance in ("j30/j301_1.sm", "j60/j601_1.sm", "j120/j1201_1.sm")
        s = ado_start(instance, SEED, true, 1, 0.8); ado_advance!(s, B)
        @assert s.schedules <= B + 2
        println(instance, " makespan=", Int(s.bestSolution.makespan), " schedules=", s.schedules)
    end
end

reverse_name = "../../../experiments/runs/bidir/rev/j30/j301_1.sm"
for (label, pert, k, l) in (("V5", 0, 0, 0), ("V6", 1, 3, 16))
    MODE == lowercase(label) || continue
    s, side, elite, n, rule, back, tie = ado_gen_start("j30/j301_1.sm", reverse_name, SEED, 1000010, 3, "hyb")
    counters = ado_ils_run!(s, side, B, elite, n, rule, back, 500, 300, pert, k, l, 7919 * SEED + 13, 1)
    @assert s.schedules <= B + 2
    @assert check_elite("j30/j301_1.sm", elite)
    println(label, " makespan=", minimum(elite.mk), " schedules=", s.schedules, " counters=", counters, " feasible=true")
end
println("SMOKE_OK")
