using Lux
using HybridModellingExperiments
using HybridModelling
import HybridModellingExperiments: Model3SP, HybridGrowthRateModel, LuxBackend, MCMCBackend, InferICs,
                                   run_simulations, LogMSELoss, save_results
import HybridModellingExperiments: SerialMode, ParallelMode
import OrdinaryDiffEqTsit5
import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
import ADTypes: AutoZygote, AutoForwardDiff
import Optimisers: Adam
import Bijectors

using Random
using JLD2
using DataFrames
import Distributions: Uniform

mode = SerialMode()
const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])
callback(l, m, p, s) = l

fixed_params = (alg = Tsit5(),
    adtype = AutoZygote(),
    abstol = 1e-4,
    reltol = 1e-4,
    tsteps,
    lr = 5e-3,
    verbose = false,
    maxiters = 50_000,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    rng = Random.MersenneTwister(1234),
    batchsize = 10,
    n_epochs = 1,
    verbose_frequency = Inf,
    loss_fn = LogMSELoss(),
    forecast_length = 10,
    model = HybridFuncRespModel()
)

function init(
        model::HybridFuncRespModel,
        ::LuxBackend;
        alg,
        abstol,
        reltol,
        sensealg,
        maxiters,
        p_true,
        perturb = 1.0f0,
        rng,
        kwargs...
)
    distrib_param = NamedTuple([dp => Product([Uniform(
                                                   sort([(1.0f0 - perturb / 2.0f0) * k,
                                                   (1.0f0 + perturb / 2.0f0) * k])...
                                               ) for k in p_true[dp]])
                                for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(
        NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    )

    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    parameters = ParameterLayer(; constraint = Constraint(p_transform), init_value = p_init)
    functional_response =  Lux.Chain(Lux.Dense(2, HlSize, NNlib.tanh),
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, 2))
    lux_model = ODEModel((; parameters, functional_response), model; alg, abstol, reltol, sensealg, maxiters)

    return (; lux_model)
end

function generate_data(model::Model3SP; alg, abstol, reltol, tspan, tsteps, rng, s, kwargs...)
    p_true = p_true = (H = [1.24, 2.5],
                        q = [4.98, 0.8],
                        r = [1.0, -0.4, -0.08],
                        A = [1.0],
                        s)

    u0_true = [0.5,0.8,0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, (; A = p_true.A, r = p_true.r)  # only estimating A and r in hybrid model
end

function create_simulation_parameters()
    segmentsizes = floor.(Int, exp.(range(log(2), log(100), length = 6)))
    ss = exp.(range(log(8f-1), log(100f-1), length = 5)))
    nruns = 10
    ic_estims = [
        InferICs(true,
            Constraint(Bijectors.NamedTransform((;
                u0 = Bijectors.bijector(Uniform(1e-3, 5e0)))))),
        InferICs(false)]
    noises = [0.2]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims, s in ss,
        noise in noises

        optim_backend = LuxBackend(Adam(fixed_params.lr),
            fixed_params.n_epochs,
            fixed_params.adtype,
            fixed_params.loss_fn;
            fixed_params.verbose_frequency,
            callback)

        data, p_true = generate_data(model; tspan, s, fixed_params...)

        varying_params = (; segmentsize,
            optim_backend,
            experimental_setup = infer_ic,
            model,
            noise,
            data,
            p_true)
        push!(pars_arr, varying_params)
    end
    return shuffle!(fixed_params.rng, pars_arr)
end

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)
