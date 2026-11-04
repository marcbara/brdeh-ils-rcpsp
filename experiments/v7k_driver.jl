# v7k_driver.jl — V7sb with a different keep probability in the BR-DEH phase.
# Leaves the frozen v7_driver.jl / v7_weighted.jl untouched.
#
#   julia --project=<env> experiments/v7k_driver.jl <instance_list> <seeds a:b> <arms,comma> <budget> <out_prefix>
#
# Extra arm: "V7sbk1" = V7sb with policyKeepProb = 1.0 (always the favourite weight,
# never the 20 % random switch). Any other arm runs exactly as in v7_driver.jl.
# Output format is the same as v7_driver.jl.

const _V7K_ARGS = copy(ARGS)
empty!(ARGS)                                   # keep v7_driver.jl from running its own main
include(joinpath(@__DIR__, "v7_driver.jl"))

const V7K_KEEP = Dict("V7sbk1" => 1.0)

function v7k_run(inst::String, rev::String, si::Int, arm::String, budget::Int)
    haskey(V7K_KEEP, arm) || return v7_run(inst, rev, si, arm, budget)
    seed = SEEDS[si]; tieseed = 1_000_003 * si + 7; ilsseed = 7919 * seed + 13
    s, side, e, n, ridx, back, tie = v7_gen_start(inst, rev, seed, tieseed, 3, "w11")
    s.policyKeepProb = V7K_KEEP[arm]
    n = Int(n); nreal = n - 2
    k = max(2, round(Int, 0.1 * nreal)); L = max(1, round(Int, 0.5 * n))
    c, wfix, uses, impr = v7_ils_run!(s, side, budget, e, n, Int(ridx), Bool(back), S_STALL, CAP,
                                      k, L, ilsseed, 1, false)   # ILS as V7sb: uniform weight sampling
    return s, e, c, wfix, uses, impr, Int(ridx), Bool(back)
end

function main_k(args)
    list, seedspec, armspec, budgetspec, prefix = args
    insts = filter(!isempty, strip.(readlines(list)))
    a, b = parse.(Int, split(seedspec, ":"))
    arms = String.(split(armspec, ","))
    budget = parse(Int, budgetspec)
    let rev = write_reversed("j30/j301_1.sm")
        for arm in unique(vcat(arms, ["V6"]))
            v7k_run("j30/j301_1.sm", rev, 1, arm, 600)
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
                    s, e, c, wfix, uses, impr, r0, back = v7k_run(String(inst), rev, si, arm, budget)
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
                println(stderr, "[v7k] $ni/$(length(insts)) $inst")
            end
        end
    end
end

isempty(_V7K_ARGS) || main_k(_V7K_ARGS)
