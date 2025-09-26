import Distributed: @everywhere
import HybridDynamicModelExperiments: setup_distributed_environment, DistributedMode
setup_distributed_environment()

@everywhere begin 
    using Lux
    using HybridDynamicModelExperiments
    using HybridDynamicModels
    import HybridDynamicModelExperiments: Model3SP, Model5SP, Model7SP, SGDBackend, MCSamplingBackend, InferICs,
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

    const callback(l, epoch, ts) = nothing
end

function generate_data(model::Model3SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (H = [1.24, 2.5],
                q = [4.98, 0.8],
                r = [1.0, -0.4, -0.08],
                A = [1.0])

    u0_true = [0.77, 0.060, 0.945]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function create_simulation_parameters()
    segmentsizes = [8]
    nruns = 5
    models = [Model3SP()]
    ic_estims = [
        InferICs(true,
            NamedTupleConstraint((;
                u0 = BoxConstraint([1e-3], [5e0])))),
        InferICs(false)]
    noises = [0.2, 0.4]
    perturbs = [0.5, 1.0]
    lrs = [1e-2]
    shifts = [2, 4, 6, 8]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims, model in models,
        noise in noises, perturb in perturbs, lr in lrs, shift in shifts

        optim_backend = SGDBackend(Adam(lr),
            fixed_params.n_epochs,
            fixed_params.adtype,
            fixed_params.loss_fn,
            callback)

        data, p_true = generate_data(model; tspan, fixed_params...)

        varying_params = (; segmentsize,
            optim_backend,
            experimental_setup = infer_ic,
            model,
            noise,
            data,
            p_true,
            perturb,
            shift)
        push!(pars_arr, varying_params)
    end
    return shuffle!(fixed_params.rng, pars_arr)
end

mode = DistributedMode()
const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])

fixed_params = (alg = Tsit5(),
    adtype = AutoZygote(),
    abstol = 1e-4,
    reltol = 1e-4,
    tsteps,
    verbose = false,
    maxiters = 50_000,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    rng = Random.MersenneTwister(1234),
    batchsize = 10,
    n_epochs = 3000,
    loss_fn = LogMSELoss(),
    forecast_length = 10,
    luxtype = Lux.f64
)

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)
