# inputs.jl

@with_kw struct Test
    instanceName :: String
    maxTime :: Int64
    maxIter :: Int64
    distCrit :: String
    betaMin :: Float64
    betaMax :: Float64
    distCand :: String
    betaMin2 :: Float64
    betaMax2 :: Float64
    seed :: Int64
    shortSim :: Int64
    longSim :: Int64
    varLevel :: Float64
    execType :: Int64
    TYPE_CRITERIA :: Int64 = 0
    TYPE_CANDIDATE :: Int64 = 1
end

@with_kw struct Inputs
    name :: String
    nTasks :: Int64
    nResources :: Int64
    timeHorizon :: Int64
    sumResources :: Int64
    tasks :: Vector{Task}
    resources :: Vector{Resource}
end

function readTests(fileName)
    tests = []
    open("tests/" * fileName) do f
        _ = readline(f)
        for line in readlines(f)
            tokens = split(line, "\t")
            if !occursin("#", tokens[1])
                # Parse test line tokens
                instanceName = tokens[1]
                maxTime = parse(Int64, tokens[2])
                maxIter = parse(Int64, tokens[3])
                distCrit = tokens[4]
                betaMin = parse(Float64, tokens[5])
                betaMax = parse(Float64, tokens[6])
                distCand = tokens[7]
                betaMin2 = parse(Float64, tokens[8])
                betaMax2 = parse(Float64, tokens[9])
                seed = parse(Int64, tokens[10])
                shortSim = parse(Int64, tokens[11])
                longSim = parse(Int64, tokens[12])
                varLevel = parse(Float64, tokens[13])
                execType = parse(Int64, tokens[14])
                # Add constructed test case to tests list
                test = Test(instanceName = instanceName, maxTime = maxTime, maxIter = maxIter, distCrit = distCrit, betaMin = betaMin, betaMax = betaMax, distCand = distCand, betaMin2 = betaMin2, betaMax2 = betaMax2, seed = seed, shortSim = shortSim, longSim = longSim, varLevel = varLevel, execType = execType)
                push!(tests, test)
            end
        end
    end
    return tests
end

function computePredecessors(tasks, task, visited)
    if task.id in visited
        return
    else
        push!(visited, task.id)
        for (predecessor_id, _) in task.predecessors
            predecessor = tasks[predecessor_id]
            computePredecessors(tasks, predecessor, visited)
        end
    end
end

function computeSuccessors(tasks, task, visited)
    if task.id in visited
        return
    else
        push!(visited, task.id)
        for (successor_id, _) in task.successors
            successor = tasks[successor_id]
            computeSuccessors(tasks, successor, visited)
        end
    end
end

function readInputs(instanceName)
    lines = readlines("inputs/" * instanceName)

    n_jobs = parse(Int64, split(lines[6])[end])
    n_resources = parse(Int64, split(lines[9])[end - 1])

    # Create list of task objects
    time_horizon = 0
    tasks = Vector{Task}()
    for i in 1:n_jobs
        parts = parse.(Int64, split(lines[18 + i]))

        # Successors
        successors = OrderedDict{Int64, Float64}()
        for j in parts[4:end]
            successors[j] = 0
        end

        parts = parse.(Int64, split(lines[18 + n_jobs + 4 + i]))
        duration = parts[3]
        resources = OrderedDict{Int64, Int64}()
        for (r, val) in enumerate(parts[4:end])
            if val > 0
                resources[r] = val
            end
        end

        # OrderedDict placeholder (overwritten below); newer OrderedCollections
        # no longer converts a plain Dict() implicitly
        task = Task(id = i, duration = duration, predecessors = OrderedDict{Int64, Float64}(), successors = successors, resources = resources)
        push!(tasks, task)

        time_horizon += duration
    end

    # Create list of resource objects
    sum_resources = 0
    resources = Vector{Resource}()
    parts = parse.(Int64, split(lines[18 + n_jobs + 4 + n_jobs + 4]))
    for r in 1:n_resources
        resource = Resource(id = r, units = parts[r])
        sum_resources += resource.units
        push!(resources, resource)
    end

    # Compute basic information for input tasks
    for task in tasks
        predecessors = OrderedDict{Int64, Float64}()
        for other_task in tasks
            if task.id in keys(other_task.successors)
                predecessors[other_task.id] = 0
            end
        end
        task.predecessors = predecessors
        task.sum_resources = sum(values(task.resources))
        task.max_percentage_resources = maximum([units / resources[resource_id].units for (resource_id, units) in task.resources], init=0)  
        task.sum_percentage_resources = maximum([units / sum_resources for (_, units) in task.resources], init=0)
    end

    # # Compute the complete predecessors and successors for each task
    # for task in tasks
    #     predecessors = Set{Int64}()
    #     computePredecessors(tasks, task, predecessors)
    #     delete!(predecessors, task.id)

    #     successors = Set{Int64}()
    #     computeSuccessors(tasks, task, successors)
    #     delete!(successors, task.id)

    #     task.full_predecessors = predecessors
    #     task.full_successors = successors
    # end

    inputs = Inputs(instanceName, length(tasks), length(resources), time_horizon, sum_resources, tasks, resources)
    return inputs
end
