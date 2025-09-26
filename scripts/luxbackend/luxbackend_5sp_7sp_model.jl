using Lux
using HybridDynamicModelExperiments
using HybridDynamicModels
import HybridDynamicModelExperiments: Model5SP, Model7SP, SGDBackend, MCSamplingBackend, InferICs,
                                   run_simulations, LogMSELoss, save_results
import HybridDynamicModelExperiments: SerialMode, ParallelMode
import OrdinaryDiffEqTsit5: Tsit5
import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
import ADTypes: AutoZygote, AutoForwardDiff
import Optimisers: Adam
import Bijectors

using Random
using JLD2
using DataFrames
import Distributions: Uniform

mode = ParallelMode()
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
    n_epochs = 3000,
    verbose_frequency = Inf,
    loss_fn = LogMSELoss(),
    forecast_length = 10,
    perturb = 1e0
)

function generate_data(model::Model5SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (ω = [0.2],
        H = [2.89855, 7.35294, 2.89855, 7.35294],
        q = [1.38, 0.272, 1.38, 0.272],
        r = [1.0, -0.15, -0.08, 1.0, -0.15],
        A = [1.0, 1.0])

    u0_true = [0.77, 0.060, 0.945, 0.467, 0.18]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function generate_data(model::Model7SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (ω = [0.2],
        H = [2.89855, 7.35294, 8.0, 2.89855, 7.35294, 12.0],
        q = [1.38, 0.272, 1e-1, 1.38, 0.272, 5e-2],
        r = [1.0, -0.15, -0.08, 1.0, -0.15, -0.01, -0.005],
        A = [1.0, 1.0])

    u0_true = [0.77, 0.060, 0.945, 0.467, 0.18, 0.14, 0.18]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function create_simulation_parameters()
    segment_lengths = floor.(Int, exp.(range(log(2), log(100), length = 6)))
    nruns = 10
    models = [Model5SP(), Model7SP()]
    ic_estims = [
        InferICs(true,
            Constraint(Bijectors.NamedTransform((;
                u0 = Bijectors.bijector(Uniform(1e-3, 5e0)))))),
        InferICs(false)]
    noises = [0.2]

    pars_arr = []
    for segment_length in segment_lengths, run in 1:nruns, infer_ic in ic_estims, model in models,
        noise in noises

        optim_backend = SGDBackend(Adam(fixed_params.lr),
            fixed_params.n_epochs,
            fixed_params.adtype,
            fixed_params.loss_fn;
            fixed_params.verbose_frequency,
            callback)

        data, p_true = generate_data(model; tspan, fixed_params...)

        varying_params = (; segment_length,
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
