#=
Utilities to run inference simulations.
=#

using UnPack
using Bijectors, ComponentArrays, Distributions, Graphs, JLD2
using LinearAlgebra, OrdinaryDiffEq, OptimizationOptimisers, OptimizationOptimJL
using PiecewiseInference, Pkg, SciMLSensitivity, SparseArrays
using Statistics

# Initialize parameters and setup constraints
function initialize_params_and_constraints(model, data, perturb_param)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([T((1-perturb_param/2) * k), T((1+perturb_param/2)* k)])...) for k in p_true[dp]]) for dp in keys(p_true)])
    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])
    return ComponentArray(p_init) .|> eltype(data), distrib_param, p_bij, u0_bij
end

# simulation function, preprocess before train, train, and postprocess
function simu(optim_backend, experimental_setup; segmentsize, batchsize, shift=nothing, noise, data, tsteps, kwargs...)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()
    isnothing(shift) && shift = segmentsize - 1

    data_w_noise = generate_noisy_data(data, noise)
    dataloader = SegmentedTimeSeries((data_w_noise, tsteps); segmentsize, batchsize, shift)

    println("Launching simulations for segmentsize = $segmentsize, noise = $noise")

    stats = @timed train(optim_backend, experimental_setup; dataloader, kwargs)

    res = stats.value
    p_trained = get_parameter_values(res)
    err = median([median(abs.(p_trained[k] - model.mp.p[k]) ./ model.mp.p[k]) for k in keys(p_trained)])
    l = get_loss(res)

    return (;segmentsize, 
            batchsize, 
            noise, 
            med_par_err = err, 
            loss = l, 
            time = stats.time, 
            res, 
            adtype = typeof(adtype), 
            optim_backend = nameof(optim_backend))
end