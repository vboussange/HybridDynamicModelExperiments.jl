using UnPack
using Bijectors, ComponentArrays, Distributions, Graphs, JLD2
using LinearAlgebra, OrdinaryDiffEq, OptimizationOptimisers, OptimizationOptimJL
using PiecewiseInference, Pkg, SciMLSensitivity, SparseArrays
using Statistics

include("hybrid_growth_rate_model.jl")
include("loss_fn.jl")

# Main simulation function
function simu(pars, datas_arr, epochs, inference_params)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()
    
    @unpack group_size, noise, adtype, d, model, infprob, pref, noise = pars

    println("Launching simulations for group_size = $group_size, noise = $noise")

    data_w_noise = datas_arr[d]

    stats = @timed inference(infprob; group_size = group_size,
                                           data = data_w_noise, 
                                           adtype, epochs=[epochs], inference_params...)

    res = stats.value
    l = res.losses[end]

    return group_size, noise, l, stats.time, res, typeof(adtype), pref, name(model)
end