#=
Utilities to run inference simulations.
=#

using HybridModelling
using Distributions
using Bijectors

function init(model::AbstractEcosystemModel; alg, abstol, reltol, sensealg, maxiters, p_true, perturb=1f0, kwargs...)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    
    # TODO: problem with rand(Uniform), casts to Float64
    p_init = NamedTuple([k => rand(distrib_param[k])  .|> eltype(p_true[1]) for k in keys(distrib_param)])

    parameters = ParameterLayer(constraint = Constraint(p_transform), 
                                init_value = p_init)
    lux_model = ODEModel((;parameters), model; alg, abstol, reltol, sensealg, maxiters)

    return lux_model
end

# simulation function, preprocess before train, train, and postprocess
function simu(optim_backend, experimental_setup; model, p_true, segmentsize, batchsize, shift=nothing, noise, data, tsteps, adtype, kwargs...)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()
    if isnothing(shift)
        shift = segmentsize - 1
    end

    data_w_noise = generate_noisy_data(data, noise)
    dataloader = SegmentedTimeSeries((data_w_noise, tsteps); segmentsize, batchsize, shift)

    # Lux model initialization with biased parameters
    lux_model = init(model; p_true, kwargs...)
    println("Launching simulations for segmentsize = $segmentsize, noise = $noise, backend = $(typeof(optim_backend)), experimental_setup = $(typeof(experimental_setup))")
    # try
        stats = @timed train(optim_backend, experimental_setup; model = lux_model, dataloader, kwargs...)
        res = stats.value
        p_trained = get_parameter_values(res)
        err = median([median(abs.(p_trained[k] - p_true[k]) ./ p_true[k]) for k in keys(p_trained)])
        l = get_loss(res)
        time = stats.time
    # catch
    #     println("Error occurred during training")
    #     res = missing
    #     time = missing
    #     err = missing
    #     l = missing
    # end


    return (;segmentsize, 
            batchsize, 
            noise, 
            med_par_err = err, 
            loss = l, 
            time, 
            res, 
            adtype = typeof(adtype), 
            optim_backend = typeof(optim_backend),
            experimental_setup = typeof(experimental_setup))
end