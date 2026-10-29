# objects.jl

@with_kw mutable struct Task
    id :: Int64
    duration :: Int64
    predecessors :: OrderedDict{Int64, Float64}
    successors :: OrderedDict{Int64, Float64}
    resources :: OrderedDict{Int64, Int64}
    start_time :: Int64 = 0
    finish_time :: Int64 = 0
    sum_resources :: Int64 = 0
    max_percentage_resources :: Float64 = 0
    sum_percentage_resources :: Float64 = 0
    predecessors_to_schedule :: Int64 = 0
    # full_predecessors :: Set{Int64} = Set()
    # full_successors :: Set{Int64} = Set()
end

@with_kw mutable struct Resource
    id :: Int64
    units :: Int64
end

@with_kw mutable struct Solution
    tasks :: Vector{Task} = []
    makespan :: Float64 = 0
    time :: Float64 = 0
end
