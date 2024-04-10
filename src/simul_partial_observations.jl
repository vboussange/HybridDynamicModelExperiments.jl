using UnPack
using Bijectors, ComponentArrays, Distributions, Graphs, JLD2
using LinearAlgebra, OrdinaryDiffEq, OptimizationOptimisers, OptimizationOptimJL
using ParametricModels, PiecewiseInference, Pkg, SciMLSensitivity, SparseArrays
using Statistics

include("3sp_model.jl")
include("5sp_model.jl")
include("7sp_model.jl")
include("loss_fn.jl")


# Generate noisy data
function generate_noisy_data(data, noise)
    return data .* exp.(noise * randn(size(data)))
end

# Initialize parameters and setup constraints
function initialize_params_and_constraints(model, data, group_size)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([T(0.25 * k), T(1.75 * k)])...) for k in p_true[dp]]) for dp in keys(p_true)])
    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    distrib_u0 = Uniform(T(1e-3), T(5e0))
    u0_bij = bijector(distrib_u0)  # For initial conditions
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])

    ranges = get_ranges(;group_size, datasize= size(data,2))

    u0s_init = data[:,first.(ranges)] # u0s for consumer and prey
    u0s_init = [vcat(rand(Uniform(T(1e-3), T(1e0))), u0) for u0 in u0s_init] # adding random u0 for resource
    return ComponentArray(p_init) .|> eltype(data), u0s_init, ranges, distrib_param, p_bij, u0_bij
end

# Main simulation function
function simu(pars, data, epochs, inference_params)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()

    @unpack group_size, noise, adtype, model, batchsize = pars

    println("Launching simulations for group_size = $group_size, noise = $noise")

    data_w_noise = generate_noisy_data(data, noise)
    p_init, u0s_init, ranges, distrib_param, p_bij, u0_bij = initialize_params_and_constraints(model, data, group_size)
    loss_likelihood = LossLikelihoodPartialObs()

    infprob = InferenceProblem(model, p_init; 
                                loss_u0_prior = loss_likelihood, 
                                loss_likelihood = loss_likelihood, 
                                p_bij, u0_bij)

    stats = @timed inference(infprob; data = data_w_noise, 
                                        batchsizes = [batchsize], 
                                        adtype, 
                                        epochs, 
                                        u0s_init,
                                        ranges,
                                        inference_params...)

    res = stats.value
    p_trained = res.p_trained
    err = median([median(abs.(p_trained[k] - model.mp.p[k]) ./ model.mp.p[k]) for k in keys(p_trained)])
    l = res.losses[end]

    return group_size, batchsize, noise, err, l, stats.time, res, typeof(adtype)
end