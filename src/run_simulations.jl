#=
Utility to run simulations in parallel using Distributed.jl
=#

using DataFrames, ProgressMeter, Distributed, ComponentArrays

abstract type AbstractSimulationMode end
struct SerialMode <: AbstractSimulationMode end
struct DistributedMode <: AbstractSimulationMode end
struct ParallelMode <: AbstractSimulationMode end

# distributes simulations

function run_simulations(mode::SerialMode, varying_params; fixed_params...)
    n = length(varying_params)
    results = Vector{Any}(undef, n)
    @showprogress for i in 1:n
        kwargs = merge(NamedTuple(fixed_params), varying_params[i])
        experimental_setup, kwargs = pop(kwargs, :experimental_setup) # TODO: check if this works
        optim_backend, kwargs = pop(kwargs, :optim_backend)
        results[i] = simu(optim_backend, experimental_setup; kwargs...)
    end
    df_results = DataFrame(results)
    return df_results
end

function run_simulations(mode::DistributedMode, varying_params; fixed_params...)
    pmap_res = @showprogress pmap(1:length(varying_params)) do i
            kwargs = merge(NamedTuple(fixed_params), varying_params[i])
            experimental_setup, kwargs = pop(kwargs, :experimental_setup) # TODO: check if this works
            optim_backend, kwargs = pop(kwargs, :optim_backend)
            simulation = simu(optim_backend, experimental_setup; kwargs...)
            return simulation
    end
    df_results = DataFrame(pmap_res)
    return df_results
end


function run_simulations(mode::ParallelMode, varying_params; fixed_params...)
    n = length(varying_params)
    results = Vector{Any}(undef, n)
    prog = Progress(n)
    Threads.@threads :greedy for i in 1:n
        kwargs = merge(NamedTuple(fixed_params), varying_params[i])
        experimental_setup, kwargs = pop(kwargs, :experimental_setup)
        optim_backend, kwargs = pop(kwargs, :optim_backend)
        res = simu(optim_backend, experimental_setup; kwargs...)
        results[i] = res
        next!(prog)
    end
    return DataFrame(results)
end

function setup_distributed_environment(procs_to_add=nothing)
    if nprocs() == 1
        isnothing(procs_to_add) && (procs_to_add = isempty(ARGS) ? 0 : parse(Int, ARGS[1]))
        addprocs(procs_to_add, exeflags="--project=$(Base.active_project())")
        @everywhere @eval using HybridModellingExperiments
        println("Running script with ", nprocs(), " process(es)")
    else
        println("Running script with ", nprocs(), " process(es)\nNot adding more processes.")
    end
end
