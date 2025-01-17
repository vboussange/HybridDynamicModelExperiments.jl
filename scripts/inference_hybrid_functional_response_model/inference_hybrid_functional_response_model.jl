#!/usr/bin/env julia

# -------------------------------------------------------------------------
#     Inference Simulation on a 3-Species Model using HybridFuncRespModel
# -------------------------------------------------------------------------

using JLD2
using DataFrames
using Random
using Dates
using ProgressMeter
using ComponentArrays
import OrdinaryDiffEq: Tsit5
using OptimizationOptimisers
using PiecewiseInference
using SciMLSensitivity
using Distributions
using Bijectors

Random.seed!(5)

include("../../src/utils.jl")
include("../../src/3sp_model.jl")
include("../../src/hybrid_functional_response_model.jl")
include("../../src/loss_fn.jl")

"""
    init()
Initialize non-neural network parameters for inference and bijectors.
"""
function init()
    T = eltype(P_TRUE)
    distrib_param_arr = Pair{Symbol, Any}[]

    for dp in keys(P_TRUE)
        dp == :p_nn && continue
        pair = dp => Product([Uniform(sort(T[(1f0-PERTURB/2f0) * k, (1f0+PERTURB/2f0) * k])...) for k in P_TRUE[dp]])
        push!(distrib_param_arr, pair)
    end
    pair_nn = :p_nn => Uniform(-Inf, Inf)
    push!(distrib_param_arr, pair_nn)

    distrib_param = NamedTuple(distrib_param_arr)

    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param) if k !== :p_nn]) |> ComponentArray

    return p_init, p_bij, u0_bij
end

function build_simulation_configs(;
    group_sizes = [11],
    noises      = [0.1],
    nruns       = 10
)
    configs = []
    for gsize in group_sizes
        for noise in noises
            for run_id in 1:nruns
                p_init, p_bij, u0_bij = init()
                # need to inform u0 to inform number of state variables
                model = HybridFuncRespModel(ModelParams(p=p_init, u0=zeros(3); SOLVER_PARAMS...))

                # Inference problem
                infprob = InferenceProblem(
                    model,
                    model.mp.p;
                    loss_u0_prior   = LOSS_LIKELIHOOD,
                    loss_likelihood = LOSS_LIKELIHOOD,
                    p_bij           = p_bij,
                    u0_bij          = u0_bij
                )

                push!(configs, (
                    group_size = gsize,
                    noise      = noise,
                    run_id     = run_id,
                    model      = model,
                    infprob    = infprob
                ))
            end
        end
    end
    return configs
end

function run_single_inference(config, data, epochs, inference_params)
    GC.gc()  # occasionally trigger garbage collection

    group_size = config.group_size
    noise      = config.noise
    run_id     = config.run_id
    model      = config.model
    infprob    = config.infprob

    println("Running inference for group_size = $group_size, noise = $noise, run_id = $run_id")

    # Add noise if needed, or just use data as is
    noisy_data = data .* exp.(noise * randn(size(data)))
    stats = @timed inference(
        infprob;
        tsteps = TSTEPS,
        group_size = group_size,
        data       = noisy_data,  # or data_noisy
        epochs     = epochs,
        inference_params...
    )

    result = stats.value
    final_loss = result.losses[end]
    return (group_size, noise, final_loss, stats.time, result, run_id, name(model))
end

P_TRUE = ComponentArray(
    H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0]
)
TSTEPS = range(500.0, step=4, length=100)
LOSS_LIKELIHOOD = LossLikelihood()
SOLVER_PARAMS = (;alg = Tsit5(),
                abstol = 1e-4,
                reltol = 1e-4,
                sensealg= BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                maxiters= 50_000,
                verbose = false)
PERTURB = 0.5
U0_TRUE = [0.77, 0.060, 0.945]

epochs = [5000]
tspan = (0.0, last(tsteps))

inference_params = (;optimizers = [OptimizationOptimisers.Adam(2e-2)],
                    verbose_loss = true,
                    info_per_its = 10,
                    multi_threading = false
                    )

true_model = Model3SP(ModelParams(;u0 = U0_TRUE,
                                p = P_TRUE, 
                                tspan = tspan,
                                saveat = tsteps,
                                SOLVER_PARAMS...))

synthetic_data = simulate(true_model, p = P_TRUE) |> Array

configs = build_simulation_configs()

results_df = DataFrame(
    group_size = Int[],
    noise      = Float64[],
    loss       = Float64[],
    time       = Float64[],
    res        = Any[],
    run_id     = Int[],
    model_name = String[],
)

for config in configs
    row_tuple = run_single_inference(config, synthetic_data, epochs, inference_params)
    push!(results_df, row_tuple)
end

save_results("simple_inference_3sp.jld2"; results=results_df)
println("Inference completed. Results saved.")

