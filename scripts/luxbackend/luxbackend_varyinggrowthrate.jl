import Distributed: @everywhere
import HybridDynamicModelExperiments: setup_distributed_environment, HybridGrowthRateModel
setup_distributed_environment()

@everywhere begin
    using Lux
    using HybridDynamicModels
    using HybridDynamicModelExperiments
    import HybridDynamicModelExperiments: VaryingGrowthRateModel, Model3SP, AbstractEcosystemModel, SGDBackend, InferICs, run_simulations, LogMSELoss, save_results
    import HybridDynamicModelExperiments: SerialMode, ParallelMode, DistributedMode
    import OrdinaryDiffEqTsit5: Tsit5
    import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
    import ADTypes: AutoZygote, AutoForwardDiff
    import Optimisers: AdamW
    import Bijectors

    using Random
    using JLD2
    using DataFrames
    import Distributions: Uniform, product_distribution
    import NNlib

    const rng = Random.MersenneTwister(1234)
    const callback(l, epoch, ts) = nothing

end

function generate_data(
        ::AbstractEcosystemModel; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [1.0])

    u0_true = [0.5, 0.8, 0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), VaryingGrowthRateModel(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function generate_data(
        ::Model3SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [0.8])

    u0_true = [0.5, 0.8, 0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), VaryingGrowthRateModel(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    p_init = (; H = [1.24, 2.5],
                q = [4.98, 0.8],
                r = [1.0, -0.4, -0.08],
                A = [1.0])

    return synthetic_data, p_init # not estimating s in Model3SP
end

function create_simulation_parameters()
    segment_lengths = [4]
    nruns = 5
    ic_estims = [
        InferICs(true,
            NamedTupleConstraint((;
                u0 = BoxConstraint([1e-3], [5e0]))))]
    noises = [0.2, 0.4]
    weight_decays = [1e-5]
    perturbs = [1.0]
    lrs = [1e-2]
    batchsizes = [10]
    models = [VaryingGrowthRateModel(), Model3SP()]

    pars_arr = []
    for segment_length in segment_lengths, run in 1:nruns, infer_ic in ic_estims,
        noise in noises, weight_decay in weight_decays, perturb in perturbs, lr in lrs,
        batchsize in batchsizes, model in models

        optim_backend = SGDBackend(AdamW(; eta = lr, lambda = weight_decay),
            n_epochs,
            adtype,
            loss_fn,
            callback)

        data, p_true = generate_data(model; tspan, fixed_params...)

        varying_params = (; segment_length,
            optim_backend,
            experimental_setup = infer_ic,
            noise,
            data,
            p_true,
            weight_decay,
            perturb,
            batchsize,
            model)
        push!(pars_arr, varying_params)
    end
    return shuffle!(rng, pars_arr)
end

const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])
const adtype = AutoZygote()
const loss_fn = LogMSELoss()
const n_epochs = 3000

mode = DistributedMode()

fixed_params = (alg = Tsit5(),
    abstol = 1e-4,
    reltol = 1e-4,
    tsteps,
    verbose = false,
    maxiters = 50_000,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    batchsize = 10,
    forecast_length = 10,
    rng,
    luxtype = Lux.f64,
)

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)
save_results(string(@__FILE__); results)
