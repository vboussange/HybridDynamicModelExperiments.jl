#=
Running inference simulations with `Model3SP` and
`HybridFuncRespModel`.
=#

cd(@__DIR__)
using JLD2
using DataFrames
using Random
using Dates
using ProgressMeter
using ComponentArrays
import OrdinaryDiffEqTsit5: Tsit5
using OptimizationOptimisers
using PiecewiseInference
using SciMLSensitivity
using Distributions
using Bijectors

Random.seed!(2)

include("../../src/utils.jl")
include("../../src/3sp_model.jl")
include("../../src/hybrid_functional_response_model.jl")
include("../../src/loss_fn.jl")

SYNTHETIC_DATA_PARAMS = (;p_true = ComponentArray(
                                                H = [1.24, 2.5],
                                                q = [4.98, 0.8],
                                                r = [1.0, -0.4, -0.08],
                                                A = [1.0]),
                        tsteps =  range(500.0, step=4, length=100),
                        solver_params = (;alg = Tsit5(),
                                        abstol = 1e-4,
                                        reltol = 1e-4,
                                        sensealg= BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                                        maxiters= 50_000,
                                        verbose = false),
                        perturb = 0.5,
                        u0_true = [0.77, 0.060, 0.945],)

SIMULATION_CONFIG = (;group_sizes = [5],
                    noises      = [0.1],
                    nruns       = 10,)

INFERENCE_PARAMS = (;optimizers = [OptimizationOptimisers.Adam(1e-2)],
                    verbose_loss = true,
                    info_per_its = 100,
                    multi_threading = false,
                    epochs = [5000],
                    loss_likelihood =  LossLikelihood(),
                    )

"""
    init()
Initialize non-neural network parameters for inference and bijectors.
"""
function init()
    T = eltype(SYNTHETIC_DATA_PARAMS.p_true)
    distrib_param_arr = Pair{Symbol, Any}[]

    for dp in [:r, :A]
        dp == :p_nn && continue
        pair = dp => Product([Uniform(sort(T[(1f0-SYNTHETIC_DATA_PARAMS.perturb/2f0) * k, (1f0+SYNTHETIC_DATA_PARAMS.perturb/2f0) * k])...) for k in SYNTHETIC_DATA_PARAMS.p_true[dp]])
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

function build_simulation_configs()
    configs = []
    for gsize in SIMULATION_CONFIG.group_sizes
        for noise in SIMULATION_CONFIG.noises
            for run_id in 1:SIMULATION_CONFIG.nruns
                p_init, p_bij, u0_bij = init()
                # need to inform u0 to inform number of state variables
                model = HybridFuncRespModel(ModelParams(;p=p_init, 
                                                        u0=zeros(Float32, 3),
                                                        saveat = SYNTHETIC_DATA_PARAMS.tsteps,
                                                        SYNTHETIC_DATA_PARAMS.solver_params...),
                                                        seed=run_id)

                # Inference problem
                infprob = InferenceProblem(
                    model,
                    model.mp.p;
                    loss_u0_prior   = INFERENCE_PARAMS.loss_likelihood,
                    loss_likelihood = INFERENCE_PARAMS.loss_likelihood,
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

function run_single_inference(config, data, inference_params)
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
        tsteps = SYNTHETIC_DATA_PARAMS.tsteps,
        group_size = group_size,
        data = noisy_data,  # or data_noisy
        optimizers = inference_params.optimizers,
        verbose_loss = inference_params.verbose_loss,
        info_per_its = inference_params.info_per_its,
        epochs = inference_params.epochs,
        multi_threading = inference_params.multi_threading,
        adtype = Optimization.AutoZygote()
    )

    result = stats.value
    final_loss = result.losses[end]
    return (group_size, noise, final_loss, stats.time, result, run_id, name(model))
end

true_model = Model3SP(ModelParams(;u0 = SYNTHETIC_DATA_PARAMS.u0_true,
                                p = SYNTHETIC_DATA_PARAMS.p_true, 
                                tspan = (0.0, last(SYNTHETIC_DATA_PARAMS.tsteps)),
                                saveat = SYNTHETIC_DATA_PARAMS.tsteps,
                                SYNTHETIC_DATA_PARAMS.solver_params...))

synthetic_data = simulate(true_model) |> Array

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
    row_tuple = run_single_inference(config, synthetic_data, INFERENCE_PARAMS)
    push!(results_df, row_tuple)
end

save_results(@__FILE__; results=results_df, synthetic_data, p_true = SYNTHETIC_DATA_PARAMS.p_true)