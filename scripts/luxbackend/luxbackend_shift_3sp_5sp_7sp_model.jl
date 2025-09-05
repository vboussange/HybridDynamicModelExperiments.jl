import Distributed: @everywhere
import HybridModellingExperiments: setup_distributed_environment
setup_distributed_environment(4)

@everywhere begin 
    using Lux
    using HybridModellingExperiments
    using HybridModelling
    import HybridModellingExperiments: Model3SP, Model5SP, Model7SP, LuxBackend, MCMCBackend, InferICs,
                                    run_simulations, LogMSELoss, save_results
    import HybridModellingExperiments: SerialMode, ParallelMode
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
end

function create_simulation_parameters()
    segmentsizes = [8]
    nruns = 5
    models = [Model3SP()]
    ic_estims = [
        InferICs(true,
            Constraint(Bijectors.NamedTransform((;
                u0 = Bijectors.bijector(Uniform(1e-3, 5e0)))))),
        InferICs(false)]
    noises = [0.2, 0.4]
    perturbs = [1.0]
    lrs = [1e-2]
    shifts = [2, 4, 6, 8]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims, model in models,
        noise in noises, perturb in perturbs, shift in shifts, lr in lrs

        optim_backend = LuxBackend(Adam(lr),
            fixed_params.n_epochs,
            fixed_params.adtype,
            fixed_params.loss_fn;
            fixed_params.verbose_frequency,
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
    perturb = 1e0
)

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters[1:10]; fixed_params...)

save_results(string(@__FILE__); results)
