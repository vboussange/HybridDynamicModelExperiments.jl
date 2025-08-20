#=
Utility to run simulations in parallel using Distributed.jl
=#

using DataFrames, ProgressMeter, Distributed, ComponentArrays

# distributes simulations
function run_simulations(varying_params; fixed_params...)
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

function setup_distributed_environment()
    procs_to_add = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
    addprocs(procs_to_add, exeflags="--project=$(Base.active_project())")
    @everywhere @eval using HybridModellingBenchmark
    println("Running script with ", nprocs(), " process(es)")
end
