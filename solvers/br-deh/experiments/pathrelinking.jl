# pathrelinking.jl — b3: the path relinking procedure described in the paper
# (sec. 4, following Ranjbar & Kianfar) that is absent from the repo.
#
# Elite solutions are stored as lightweight (order, makespan) tuples — id
# permutations — because Solution objects alias the shared Task structs,
# which every construction mutates.
#
# Underspecifications resolved here (documented choices):
#   - elite list management: unbounded, one entry per strict improvement
#     of the incumbent (Algorithm 1 pushes every new best)
#   - precedence repair: move a task after its latest-positioned
#     predecessor, repeat to fixpoint
#   - after a repair the target position may no longer hold the guiding
#     task; we proceed to the next position (paper is ambiguous)
#   - evaluation: decode (serial SGS, earliest feasible start) after every
#     effective swap; each decode counts as one schedule

function repairPrecedence!(order, inputs)
    changed = true
    while changed
        changed = false
        for i in 2:length(order)
            task = inputs.tasks[order[i]]
            maxpred = 0
            for (pid, _) in task.predecessors
                p = findfirst(==(pid), order)
                maxpred = max(maxpred, p)
            end
            if maxpred > i
                id = order[i]
                deleteat!(order, i)
                insert!(order, maxpred, id)
                changed = true
                break
            end
        end
    end
    return order
end

# Serial SGS decode of an id permutation: earliest resource- and
# precedence-feasible start for each task, no upper bound on the horizon.
# (forwardPass cannot be used here: it caps the search at the inherited
# start_time.) makespan of any left-shifted schedule is bounded by
# timeHorizon = sum of durations, so the availability matrix always fits.
function decodeOrder!(inputs, availability_units, order, finish_times)
    fill!(availability_units, 0)
    makespan = 0
    @inbounds for id in order
        task = inputs.tasks[id]
        est = 0
        for (pid, _) in task.predecessors
            est = max(est, finish_times[pid])
        end
        t = est
        if !isempty(task.resources)
            while true
                ok = true
                for rt in t : t + task.duration - 1
                    for (r, u) in task.resources
                        if availability_units[r, rt + 1] + u > inputs.resources[r].units
                            ok = false
                            break
                        end
                    end
                    ok || break
                end
                ok && break
                t += 1
            end
            for rt in t : t + task.duration - 1, (r, u) in task.resources
                availability_units[r, rt + 1] += u
            end
        end
        finish_times[id] = t + task.duration
        makespan = max(makespan, finish_times[id])
    end
    SCHEDULE_COUNT[] += 1
    return makespan
end

# Iterate every elite pair; walk the trajectory from initiating to guiding
# by position swaps; report every strict improvement via record!(makespan).
function pathRelinkingOrders!(inputs, availability_units, elites, bestMakespan, record!)
    finish_times = Vector{Int64}(undef, inputs.nTasks)
    for i in 1:length(elites)-1, j in i+1:length(elites)
        a, b = elites[i], elites[j]
        guiding, initiating = a.makespan <= b.makespan ? (a, b) : (b, a)
        order = copy(initiating.order)
        for pos in 1:length(order)
            if order[pos] != guiding.order[pos]
                src = findfirst(==(guiding.order[pos]), order)
                order[pos], order[src] = order[src], order[pos]
                repairPrecedence!(order, inputs)
                mk = decodeOrder!(inputs, availability_units, order, finish_times)
                if mk < bestMakespan
                    bestMakespan = Float64(mk)
                    record!(bestMakespan)
                end
            end
        end
    end
    return bestMakespan
end
