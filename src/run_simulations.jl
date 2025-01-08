using DataFrames, ProgressMeter, Distributed, ComponentArrays
include("3sp_model.jl")
include("5sp_model.jl")
include("7sp_model.jl")
include("hybrid_growth_rate_model.jl")
include("hybrid_functional_response_model.jl")
include("loss_fn.jl")

function pack_simulation_parameters(;kwargs...)
    sim_params = Dict{Symbol, Any}()
    for k in kwargs
        sim_params[k[1]] = k[2]
    end
    return sim_params
end

function run_simulations(pars_arr, epochs, data; kwargs...)
    inference_params = generate_inference_params()
    pmap_res = @showprogress pmap(1:length(pars_arr)) do i
        try
            (true, simu(pars_arr[i], data, epochs, inference_params), kwargs...)
        catch e
            println("error with", pars_arr[i])
            println(e)
            (false, nothing)
        end
    end

    df_results = generate_df_results()

    for (st, row) in pmap_res
        if st 
            push!(df_results, row)
        end
    end
    return df_results
end

function setup_distributed_environment(simul_file)
    procs_to_add = isempty(ARGS) ? 0 : parse(Int, ARGS[1])
    addprocs(procs_to_add, exeflags="--project=$(Base.active_project())")
    @everywhere include(joinpath(string(@__DIR__), $simul_file))
    println("Running script with ", nprocs(), " process(es)")
end
