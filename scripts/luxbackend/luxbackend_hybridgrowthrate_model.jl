using Lux
using HybridDynamicModelExperiments
using HybridDynamicModels
import HybridDynamicModelExperiments: VaryingGrowthRateModel, HybridGrowthRateModel, Model3SP,
                                   SGDBackend, MCSamplingBackend, InferICs, run_simulations,
                                   LogMSELoss, save_results
import HybridDynamicModelExperiments: SerialMode, ParallelMode
import OrdinaryDiffEqTsit5: Tsit5
import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
import ADTypes: AutoZygote, AutoForwardDiff
import Optimisers: Adam
import Bijectors

using Random
using JLD2
using DataFrames
import Distributions: Uniform, product_distribution
import NNlib

mode = SerialMode()
const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])
const HlSize = 16
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
    n_epochs = 3000,
    verbose_frequency = Inf,
    loss_fn = LogMSELoss(),
    forecast_length = 10
)

function HybridDynamicModelExperiments.init(
        model::HybridGrowthRateModel,
        ::SGDBackend;
        alg,
        abstol,
        reltol,
        sensealg,
        maxiters,
        p_true,
        perturb = 1e0,
        rng,
        kwargs...
)
    p_true = merge(p_true, (; r = p_true.r[2:end]))
    distrib_param = NamedTuple([dp => product_distribution([Uniform(
                                                                sort([
                                                                (1e0 - perturb / 2e0) * k,
                                                                (1e0 + perturb / 2e0) * k])...
                                                            ) for k in p_true[dp]])
                                for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(
        NamedTuple([dp => Bijectors.bijector(distrib_param[dp])
                    for dp in keys(distrib_param)])
    )

    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    parameters = ParameterLayer(; constraint = Constraint(p_transform), init_value = p_init)
    growth_rate = Lux.Chain(Lux.Dense(1, HlSize, NNlib.tanh),
        Lux.Dense(HlSize, HlSize, NNlib.tanh),
        Lux.Dense(HlSize, HlSize, NNlib.tanh),
        Lux.Dense(HlSize, 1))
    lux_model = ODEModel(
        (; parameters, growth_rate), model; alg, abstol, reltol, sensealg, maxiters)

    return (; lux_model)
end

function generate_data(
        ::HybridGrowthRateModel; alg, abstol, reltol, tspan, tsteps, rng, s, kwargs...)
    p_true = (H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [s])

    u0_true = [0.5, 0.8, 0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), VaryingGrowthRateModel(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function create_simulation_parameters()
    # segment_lengths = floor.(Int, exp.(range(log(2), log(100), length = 6)))
    segment_lengths = [9]
    ss = exp.(range(log(8e-1), log(100e-1), length = 5))
    nruns = 10
    ic_estims = [
        InferICs(true,
            Constraint(Bijectors.NamedTransform((;
                u0 = Bijectors.bijector(Uniform(1e-3, 5e0)))))),
        InferICs(false)]
    noises = [0.2]
    models = [HybridGrowthRateModel(), Model3SP()]

    pars_arr = []
    for segment_length in segment_lengths, run in 1:nruns, infer_ic in ic_estims, s in ss,
        model in models,
        noise in noises

        optim_backend = SGDBackend(Adam(fixed_params.lr),
            fixed_params.n_epochs,
            fixed_params.adtype,
            fixed_params.loss_fn;
            fixed_params.verbose_frequency,
            callback)

        data, p_true = generate_data(fixed_params.model; tspan, s, fixed_params...)

        varying_params = (; segment_length,
            optim_backend,
            experimental_setup = infer_ic,
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
