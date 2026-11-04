# v7_driver.jl — run V6 and the V7 arms (see v7_weighted.jl) without Python.
#
#   julia --project=<env> experiments/v7_driver.jl <instance_list> <seeds a:b> <arms,comma> <budget> <out_prefix>
#
# Parameters are those of br_ils.run_one for arm br_r10_l5 (V6): both directions,
# S_STALL = 500, CAP = 300, window k = max(2, round(0.1 n_real)), L = max(1, round(0.5 n)),
# ILS seed 7919 * seed + 13, tie seed 1_000_003 * seed_index + 7.
# Writes <out_prefix>_runs.csv (one row per run) and <out_prefix>_traj.csv (improvement rows).

const REPO = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "harness", "solvers", "brdeh_bridge.jl"))
include(joinpath(REPO, "experiments", "escoto_hybrid_policies_90f8ab1.jl"))
include(joinpath(REPO, "experiments", "extensions.jl"))
include(joinpath(REPO, "experiments", "v7_weighted.jl"))

const S_STALL = 500
const CAP = 300

# Port of experiments/bidir.py write_reversed (same bytes; checked against the
# reverse_input_sha256 stored in the final-50k records).
function write_reversed(inst::String)
    lines = readlines(joinpath(ADO_SOLVER_ROOT, "inputs", inst))
    n = parse(Int, split(lines[6])[end])
    succ = Dict(i => [parse(Int, x) for x in split(lines[18 + i])[4:end]] for i in 1:n)
    pred = Dict(i => Int[] for i in 1:n)
    for i in 1:n, j in succ[i]
        push!(pred[j], i)
    end
    out = copy(lines)
    for new in 1:n
        old = n + 1 - new
        ns = sort([n + 1 - p for p in pred[old]])
        out[18 + new] = "$new 1 $(length(ns)) " * join(string.(ns), " ")
        parts = split(lines[18 + n + 4 + old])
        out[18 + n + 4 + new] = join([string(new); parts[2:end]], " ")
    end
    path = joinpath(REPO, "experiments", "runs", "bidir", "rev", inst)
    mkpath(dirname(path))
    write(path, join(out, "\n") * "\n")
    return "../../../experiments/runs/bidir/rev/" * inst
end

# Independent feasibility check of a start-time vector (as tests/smoke.jl check_elite).
function v7_feasible(instance, mk, starts)
    inp = cd(() -> readInputs(instance), ADO_SOLVER_ROOT)
    for task in inp.tasks, (pid, _) in task.predecessors
        starts[pid] + inp.tasks[pid].duration <= starts[task.id] || return false
    end
    maximum(starts[t.id] + t.duration for t in inp.tasks) == mk || return false
    for resource in inp.resources, t in 0:mk-1
        used = 0
        for task in inp.tasks, (rid, units) in task.resources
            rid == resource.id && starts[task.id] <= t < starts[task.id] + task.duration && (used += units)
        end
        used <= resource.units || return false
    end
    return true
end

# One run. Returns (state, elite, counters, wfix, uses, impr, init_ridx).
function v7_run(inst::String, rev::String, si::Int, arm::String, budget::Int)
    seed = SEEDS[si]; tieseed = 1_000_003 * si + 7; ilsseed = 7919 * seed + 13
    # arm => (start/tries pool, ILS weight mode, ILS weight taken from the pool rule)
    spec = Dict("V6" => ("hyb", 0, false), "V7w5" => ("hyb", 0, false),
                "V7a" => ("w11", 0, true), "V7r" => ("w11", 3, true),
                "V7b" => ("hyb", 1, false), "V7c" => ("hyb", 2, false),
                "V7s" => ("w11", 0, false), "V7h7" => ("h7", 0, false),
                "V7p21" => ("p21", 0, false), "V7p42" => ("p42", 0, false),
                "V7bp21" => ("hyb", 1, false), "V7bh7" => ("hyb", 1, false),
                "V7bp55" => ("hyb", 1, false),
                "V7sb" => ("w11", 1, false))   # V7s start + V7b ILS
    ilsKind = Dict("V7bp21" => "p21", "V7bh7" => "h7", "V7bp55" => "p55")
    kind, mode, gridIsPool = spec[arm]
    start = arm == "V6" ? ado_gen_start : v7_gen_start
    s, side, e, n, ridx, back, tie = start(inst, rev, seed, tieseed, 3, kind)
    n = Int(n); nreal = n - 2
    k = max(2, round(Int, 0.1 * nreal)); L = max(1, round(Int, 0.5 * n))
    G = length(V7_GRID)
    if arm == "V6"
        c = ado_ils_run!(s, side, budget, e, n, Int(ridx), Bool(back), S_STALL, CAP, 1, k, L, ilsseed, 1)
        return s, e, c, 0, zeros(Int, G), zeros(Int, G), Int(ridx), Bool(back)
    end
    c, wfix, uses, impr = v7_ils_run!(s, side, budget, e, n, Int(ridx), Bool(back), S_STALL, CAP,
                                      k, L, ilsseed, mode, gridIsPool; ilsKind = get(ilsKind, arm, "w11"))
    return s, e, c, wfix, uses, impr, Int(ridx), Bool(back)
end

function main(args)
    list, seedspec, armspec, budgetspec, prefix = args
    insts = filter(!isempty, strip.(readlines(list)))
    a, b = parse.(Int, split(seedspec, ":"))
    arms = String.(split(armspec, ","))
    budget = parse(Int, budgetspec)
    # warm-up so JIT does not land on the first timed run
    let rev = write_reversed("j30/j301_1.sm")
        for arm in unique(vcat(arms, ["V6"]))
            v7_run("j30/j301_1.sm", rev, 1, arm, 600)
        end
    end
    open(prefix * "_runs.csv", "w") do fr
        open(prefix * "_traj.csv", "w") do ft
            println(fr, "instance,seed_idx,arm,budget,makespan_at_cap,schedules,init_rule,init_back,ils_start,ils_iters,ils_accepted,ils_records,wfix,uses,improvements,feasible,cpu_s")
            println(ft, "instance,seed_idx,arm,schedules,makespan")
            for (ni, inst) in enumerate(insts)
                rev = write_reversed(String(inst))
                for si in a:b, arm in arms
                    t0 = time()
                    s, e, c, wfix, uses, impr, r0, back = v7_run(String(inst), rev, si, arm, budget)
                    dt = time() - t0
                    rows = [(Int(r.schedules), Int(r.makespan)) for r in s.traj if r.schedules <= budget]
                    atcap = minimum(x[2] for x in rows)
                    i = findfirst(==(atcap), e.mk)
                    ok = i !== nothing && v7_feasible(String(inst), atcap, e.st[i])
                    ok || error("feasibility failure $inst $si $arm")
                    println(fr, join([inst, si, arm, budget, atcap, s.schedules, r0, back, c[8], c[5], c[6], c[7],
                                      wfix, join(uses, " "), join(impr, " "), ok, round(dt, digits = 3)], ","))
                    for (sc, mk) in rows
                        println(ft, "$inst,$si,$arm,$sc,$mk")
                    end
                end
                flush(fr); flush(ft)
                println(stderr, "[v7] $ni/$(length(insts)) $inst")
            end
        end
    end
end

isempty(ARGS) || main(ARGS)
