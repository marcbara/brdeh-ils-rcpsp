# heuristic.jl

using Random
using Formatting
using Parameters
using Base.Threads
using Distributions
using DataStructures

include("objects.jl")
include("inputs.jl")

# Instrumentation: counts every schedule generated (DEH constructions and
# each backward/forward S-SGS pass), following the community convention of
# measuring effort in generated schedules (Hartmann & Kolisch).
const SCHEDULE_COUNT = Ref{Int64}(0)

#
# NOTE: MPJ Heuristic
#

# Tie-break policies unified in one concrete callable struct so that sort!
# specializes once instead of dynamically dispatching per comparison (the
# original closures each had a distinct type). Comparator logic identical.
struct ResourceTiePolicy
    mode :: Int8
    rng :: Xoshiro
end

@inline function (p::ResourceTiePolicy)(a, b)
    # First condition: task earliest start time (all policies share it)
    if a.start_time < b.start_time
        return true
    elseif a.start_time > b.start_time
        return false
    end
    m = p.mode
    m == Int8(1) && return rand(p.rng, Float64) < 0.5                                   # random
    m == Int8(2) && return a.sum_resources < b.sum_resources                            # low units
    m == Int8(3) && return a.sum_resources > b.sum_resources                            # high units
    m == Int8(4) && return a.max_percentage_resources < b.max_percentage_resources      # low pct
    m == Int8(5) && return a.max_percentage_resources > b.max_percentage_resources      # high pct
    m == Int8(6) && return a.sum_percentage_resources < b.sum_percentage_resources      # low total pct
    return a.sum_percentage_resources > b.sum_percentage_resources                      # high total pct
end

startTimeAndRandomResources(inputs, rng)              = ResourceTiePolicy(Int8(1), rng)
startTimeAndLowUnitResources(inputs, rng)             = ResourceTiePolicy(Int8(2), rng)
startTimeAndHighUnitResources(inputs, rng)            = ResourceTiePolicy(Int8(3), rng)
startTimeAndLowPercentageResources(inputs, rng)       = ResourceTiePolicy(Int8(4), rng)
startTimeAndHighPercentageResources(inputs, rng)      = ResourceTiePolicy(Int8(5), rng)
startTimeAndLowTotalPercentageResources(inputs, rng)  = ResourceTiePolicy(Int8(6), rng)
startTimeAndHighTotalPercentageResources(inputs, rng) = ResourceTiePolicy(Int8(7), rng)

function getRandomPosition(beta, size, rng)
    index = trunc(Int64, log(rand(rng, Float64)) / log(1 - beta))
    index = index % size
    return index + 1
end

function isSolutionFeasible(inputs, solution)
    # Check if the task dependencies are feasible
    for (i, task) in enumerate(solution.tasks)
        isPred = any(task.id in t.full_predecessors for t in solution.tasks[1:i-1])
        isSucc = any(task.id in t.full_successors for t in solution.tasks[i+1:end])
        if isPred || isSucc
            return false
        end
    end
    # Check if the resource dependencies are feasible
    availability_units = Dict(resource.id => resource.units for resource in inputs.resources)
    for i in 0 : solution.makespan
        finish_tasks = [t for t in solution.tasks if t.finish_time == i]
        for task in finish_tasks
            for (resource_id, units) in task.resources
                availability_units[resource_id] += units
            end
        end
        start_tasks = [t for t in solution.tasks if t.start_time == i]
        for task in start_tasks
            for (resource_id, units) in task.resources
                availability_units[resource_id] -= units
            end
        end
        if any(units < 0 for units in values(availability_units))
            return false
        end
    end
    return true
end

function MPJ_Heuristic(inputs, availability_times, beta, sortPolicy, isRandom, rng)
    solution = Solution()

    # Reset tasks predecessors to schedule counters
    @inbounds for task in inputs.tasks
        task.predecessors_to_schedule = length(task.predecessors)
    end
    # Reset resources availability times
    @inbounds for resource in inputs.resources
        availability_times[resource.id] .= 0
    end

    # Initialize candidate list with the tasks project start task
    candidateList = [inputs.tasks[begin]]

    while length(candidateList) != 0
        # Select candidate task from the list
        pos = isRandom ? getRandomPosition(beta, length(candidateList), rng) : 1
        task = popat!(candidateList, pos)
        task.finish_time = task.start_time + task.duration

        # Add task to solution schedule
        push!(solution.tasks, task)
        solution.makespan = max(solution.makespan, task.finish_time)

        # Update availability of assigned resources
        @inbounds for (resource_id, units) in task.resources
            finish_times = availability_times[resource_id]
            finish_times[1 : units] .= task.finish_time
            sort!(finish_times, by = x -> x, alg=InsertionSort)
        end

        # Update candidate list by including elegible tasks to be scheduled
        @inbounds for (successor_id, _) in task.successors
            successor = inputs.tasks[successor_id]
            # Decrement the scheduled count for each successor of task
            successor.predecessors_to_schedule -= 1
            if successor.predecessors_to_schedule == 0
                # Calculate earliest start time for task considering predecessor dependencies
                earliest_start_time = 0.0
                @inbounds for (pred_id, _) in successor.predecessors
                    pred_task = inputs.tasks[pred_id]
                    earliest_start_time = max(earliest_start_time, pred_task.finish_time)
                end
                successor.start_time = earliest_start_time
                push!(candidateList, successor)
            end
        end

        # Update candidate list tasks earliest start time considering resource availability
        @inbounds for cand_task in candidateList
            # Consider the time each resource units become available
            @inbounds for (resource_id, units) in cand_task.resources
                finish_times = availability_times[resource_id]
                finish_time = finish_times[units]
                # Update earliest start time considering new finish time
                cand_task.start_time = max(cand_task.start_time, finish_time)
            end
        end

        # Sort candidate list by selected policy
        sort!(candidateList, lt=sortPolicy, alg=InsertionSort)
    end

    # Assert the created solution is feasible
    # @assert isSolutionFeasible(inputs, solution)

    SCHEDULE_COUNT[] += 1
    return solution
end

# NOTE: FBI Local Search

function forwardRule(sorted_tasks)
    # Precompute id -> position once: same tie-break as the original
    # findfirst (ids are unique), without an O(n) scan per comparison
    pos = Vector{Int64}(undef, length(sorted_tasks))
    @inbounds for (i, t) in enumerate(sorted_tasks)
        pos[t.id] = i
    end
    @inline function compare_tasks(a, b)
        # Ascending start time on the immediate backward schedule
        if a.start_time < b.start_time
            return true
        elseif a.start_time > b.start_time
            return false
        else
            return pos[a.id] > pos[b.id]
        end
    end
    return compare_tasks
end

function backwardRule(sorted_tasks)
    pos = Vector{Int64}(undef, length(sorted_tasks))
    @inbounds for (i, t) in enumerate(sorted_tasks)
        pos[t.id] = i
    end
    @inline function compare_tasks(a, b)
        # Descending finish times on the preceding forward schedule
        if a.finish_time > b.finish_time
            return true
        elseif a.finish_time < b.finish_time
            return false
        else
            return pos[a.id] > pos[b.id]
        end
    end
    return compare_tasks
end

function forwardPass(inputs, availability_units, sorted_tasks)
    solution = Solution()
    solution.tasks = sorted_tasks

    # Reset resources availability units
    fill!(availability_units, 0)

    # Schedule project start task
    start_task = inputs.tasks[begin]
    start_task.start_time = 0
    start_task.finish_time = 0

    # Schedule activity list sorted tasks
    latest_finish_time = 0
    @inbounds for i in 2 : inputs.nTasks - 1
        task = sorted_tasks[i]

        # Calculate earliest start time for task considering predecessors
        earliest_start_time = 0
        @inbounds for (pred_id, _) in task.predecessors
            pred_task = inputs.tasks[pred_id]
            earliest_start_time = max(earliest_start_time, pred_task.finish_time)
        end

        # Refine finish time for task considering resource availability
        finish_time = typemax(Int64)
        @inbounds for schedule_time in earliest_start_time : task.start_time
            # Check if schedule time is resource feasible
            is_start_feasible = true
            @inbounds for resource_time in schedule_time : schedule_time + task.duration - 1
                for (r_id, units) in task.resources
                    if availability_units[r_id, resource_time + 1] + units > inputs.resources[r_id].units
                        is_start_feasible = false
                        break
                    end
                end
                is_start_feasible || break
            end
            # Update finish time considering scheduled task ending time
            if is_start_feasible
                finish_time = schedule_time + task.duration
                break
            end
        end

        # Schedule project task
        task.start_time = finish_time - task.duration
        task.finish_time = finish_time
        latest_finish_time = max(latest_finish_time, finish_time)
        
        # Update resource availability at scheduled times
        @inbounds for resource_time in task.start_time : task.finish_time - 1
            @inbounds for (resource_id, units) in task.resources
                availability_units[resource_id, resource_time + 1] += units
            end
        end
    end

    # Schedule project finish task
    end_task = inputs.tasks[end]
    end_task.start_time = latest_finish_time
    end_task.finish_time = latest_finish_time

    # Compute solution makespan
    solution.makespan = latest_finish_time

    # Assert the created solution is feasible
    # @assert isSolutionFeasible(inputs, solution)

    SCHEDULE_COUNT[] += 1
    return solution
end

function backwardPass(inputs, availability_units, sorted_tasks, time_horizon)
    solution = Solution()
    solution.tasks = sorted_tasks

    # Reset resources availability units
    fill!(availability_units, 0)

    # Schedule project end task
    end_task = inputs.tasks[end]
    end_task.start_time = time_horizon
    end_task.finish_time = time_horizon

    # Schedule activity list sorted tasks
    earliest_start_time = typemax(Int64)
    @inbounds for i in 2 : inputs.nTasks - 1
        task = sorted_tasks[i]

        # Calculate latest finish time for task considering successors
        latest_finish_time = typemax(Int64)
        @inbounds for (succ_id, _) in task.successors
            succ_task = inputs.tasks[succ_id]
            latest_finish_time = min(latest_finish_time, succ_task.start_time)
        end

        # Refine start time for task considering resource availability
        start_time = 0
        @inbounds for schedule_time in latest_finish_time : -1 : task.finish_time
            # Check if schedule time is resource feasible
            is_start_feasible = true
            @inbounds for resource_time in schedule_time - task.duration + 1 : schedule_time
                for (r_id, units) in task.resources
                    if availability_units[r_id, resource_time + 1] + units > inputs.resources[r_id].units
                        is_start_feasible = false
                        break
                    end
                end
                is_start_feasible || break
            end
            # Update start time considering scheduled task ending time
            if is_start_feasible
                start_time = schedule_time - task.duration
                break
            end
        end

        # Schedule project task
        task.start_time = start_time
        task.finish_time = start_time + task.duration
        earliest_start_time = min(earliest_start_time, start_time)

        # Update resource availability at scheduled times
        @inbounds for resource_time in task.start_time + 1 : task.finish_time
            @inbounds for (resource_id, units) in task.resources
                availability_units[resource_id, resource_time + 1] += units
            end
        end
    end

    # Schedule project start task
    start_task = inputs.tasks[begin]
    start_task.start_time = earliest_start_time
    start_task.finish_time = earliest_start_time

    # Compute solution makespan
    solution.makespan = time_horizon - earliest_start_time

    # Assert the created solution is feasible
    # reverse!(solution.tasks)
    # @assert isSolutionFeasible(inputs, solution)

    SCHEDULE_COUNT[] += 1
    return solution
end

# maxPasses: instrumentation parameter. typemax = as-coded behavior (loop
# until no improvement); 1 = as-described in the paper (single FBI pass).
function localSearch(solution, inputs, availability_units; maxPasses = typemax(Int64))
    improv = true
    passes = 0
    while improv && passes < maxPasses
        improv = false
        passes += 1

        # Iterate the forward schedule and build backward solution
        backward_tasks = sort(inputs.tasks, lt = backwardRule(solution.tasks))
        newSolution = backwardPass(inputs, availability_units, backward_tasks, solution.makespan)

        # Iterate the backward schedule and build forward solution
        forward_tasks = sort(inputs.tasks, lt = forwardRule(newSolution.tasks))
        newSolution = forwardPass(inputs, availability_units, forward_tasks)

        # Check if the new solution improves the current solution
        if newSolution.makespan < solution.makespan
            solution = newSolution
            improv = true
        end
    end

    return solution
end

#
# NOTE: BR-MS MPJ Heuristic
#

function detExecution(inputs, test, sortPolicies, rng)
    bestBeta = 0.5
    bestPolicy = startTimeAndHighUnitResources(inputs, rng)

    # Initialize resource availability times and units caches
    availability_times = Dict(r.id => zeros(r.units) for r in inputs.resources)
    availability_units = zeros(Int64, inputs.nResources, inputs.timeHorizon + 1)

    # Select initial beta parameter and sort policy
    beta = bestBeta
    sortPolicy = bestPolicy
    # Create an initial solution using a constructive heuristic approach
    initSolution = MPJ_Heuristic(inputs, availability_times, beta, sortPolicy, false, rng)
    initSolution = localSearch(initSolution, inputs, availability_units)
    bestSolution = initSolution

    # Start the multi-start process
    i = 0
    startTime = time()
    while i < test.maxIter
        # Select new beta parameter and sort policy
        beta = rand(rng, TriangularDist(0.0, 1.0, bestBeta))
        sortPolicy = rand(rng, Float64) < 0.5 ? bestPolicy : rand(rng, sortPolicies)
        # Create a new solution using a randomized constructive heuristic approach
        solution = MPJ_Heuristic(inputs, availability_times, beta, sortPolicy, true, rng)
        solution = localSearch(solution, inputs, availability_units)
        if solution.makespan < bestSolution.makespan
            bestBeta = beta
            bestPolicy = sortPolicy
            bestSolution = solution
            bestSolution.time = time() - startTime
        end
        # Update the iteration before evaluating the stopping criterion
        i += 1
    end

    return bestSolution
end

function printSolution(solution)
    # for task in solution.tasks
    # 	printfmtln("Task #{:d}: start time: {:d}, end time: {:d}", task.id, solution.status[task.id][0], solution.status[task.id][1])
    # end
    printfmtln("Makespan: {:.2f}", solution.makespan)
    printfmtln("Time: {:.2f}\n", solution.time)
end

function main()
    # Read tests from the file
    tests = readTests("test2run.txt")

    # Define seed values for generating random number generators
    seeds = [
        640942923, 697932107, 429007502, 435314368, 425074063, 529343854, 864294084, 85138537,
        310779084, 975978038, 157807440, 293871969, 530462484, 612032821, 60621784, 128299997,
        591873491, 245899912, 620347607, 660916590, 748384338, 907673553, 941867327, 898533368,
        548390311, 512855406, 990831284, 114651404, 193823050, 551855515, 443895884, 794990778,
    ]

    for test in tests
        # Create a pool of nTasks solutions and RNGs
        nTasks = 32
        poolSolutions = Vector{Solution}(undef, nTasks)
        poolRngs = [Xoshiro(seeds[i]) for i in 1 : nTasks]

        @threads for i in 1 : nTasks
            # Read inputs for the test inputs
            inputs = readInputs(test.instanceName)
            rng = poolRngs[i]

            # Define sort policies for choosing candidate tasks
            sortPolicies = [
                startTimeAndRandomResources(inputs, rng),
                startTimeAndLowUnitResources(inputs, rng),
                startTimeAndHighUnitResources(inputs, rng),
                startTimeAndLowPercentageResources(inputs, rng),
                startTimeAndHighPercentageResources(inputs, rng),
                startTimeAndLowTotalPercentageResources(inputs, rng),
                startTimeAndHighTotalPercentageResources(inputs, rng),
            ]

            # Compute the best deterministic solution
            poolSolutions[i] = detExecution(inputs, test, sortPolicies, rng)
        end

        # Find the best solution from the pool
        bestSolution = argmin(x -> x.makespan, poolSolutions)
        printfmtln("OBD {:s}", test.instanceName)
        printSolution(bestSolution)
    end
end

# Only run main() when executed as a script, not when included from a runner
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
