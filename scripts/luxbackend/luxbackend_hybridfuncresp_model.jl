import Distributed: @everywhere
import HybridModellingExperiments: setup_distributed_environment
setup_distributed_environment()

@everywhere begin 
    using Lux
    using HybridModelling
    using HybridModellingExperiments
    import HybridModellingExperiments: Model3SP, HybridFuncRespModel, LuxBackend, MCMCBackend,
                                    InferICs, run_simulations, LogMSELoss, save_results,
                                    InferICs
    import HybridModellingExperiments: SerialMode, ParallelMode, DistributedMode
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

    const tsteps = range(500e0, step = 4, length = 111)
    const tspan = (0e0, tsteps[end])
    const HlSize = 5
    const adtype = AutoZygote()
    const loss_fn = LogMSELoss()
    const verbose_frequency = Inf
    const n_epochs = 5000
    const rng = Random.MersenneTwister(1234)

    function HybridModellingExperiments.init(
            model::HybridModellingExperiments.HybridFuncRespModel,
            ::LuxBackend;
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
        distrib_param = NamedTuple([dp => product_distribution([Uniform(
                                                                    sort([
                                                                    (1e0 - perturb / 2e0) *
                                                                    k,
                                                                    (1e0 + perturb / 2e0) *
                                                                    k])...
                                                                ) for k in p_true[dp]])
                                    for dp in keys(p_true)])

        p_transform = Bijectors.NamedTransform(
            NamedTuple([dp => Bijectors.bijector(distrib_param[dp])
                        for dp in keys(distrib_param)])
        )

        p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

        parameters = ParameterLayer(; constraint = Constraint(p_transform), init_value = p_init)
        functional_response = Lux.Chain(Lux.Dense(2, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, 2))
        lux_model = ODEModel(
            (; parameters, functional_response), model; alg, abstol, reltol, sensealg, maxiters)

        return lux_model
    end
end

function generate_data(
        ::HybridFuncRespModel; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0])

    u0_true = [0.77, 0.060, 0.945]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, (; A = p_true.A, r = p_true.r)  # only estimating A and r in hybrid model
end

function create_simulation_parameters()
    segmentsizes = floor.(Int, exp.(range(log(2), log(100), length = 6)))
    nruns = 10
    ic_estims = [
        InferICs(true,
            Constraint(Bijectors.NamedTransform((;
                u0 = Bijectors.bijector(Uniform(1e-3, 5e0)))))),
        InferICs(false)]
    noises = [0.1, 0.2, 0.3]
    weight_decays = [1e-7, 1e-5, 1e-3, 1e-2]
    perturbs = [0.5, 1.0]
    lrs = [1e-3, 1e-2, 1e-1]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims,
        noise in noises, weight_decay in weight_decays, perturb in perturbs, lr in lrs

        optim_backend = LuxBackend(AdamW(eta = lr, lambda = weight_decay),
            n_epochs,
            adtype,
            loss_fn;
            verbose_frequency)

        data, p_true = generate_data(fixed_params.model; tspan, fixed_params...)

        varying_params = (; segmentsize,
            optim_backend,
            experimental_setup = infer_ic,
            noise,
            data,
            p_true,
            weight_decay,
            perturb)
        push!(pars_arr, varying_params)
    end
    return shuffle!(rng, pars_arr)
end

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
    model = HybridFuncRespModel(),
    rng,
    luxtype = Lux.f32
)

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)
