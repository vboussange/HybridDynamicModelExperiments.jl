import Distributed: @everywhere
import HybridModellingExperiments: setup_distributed_environment, HybridFuncRespModel
setup_distributed_environment()

@everywhere begin
    using Lux
    using HybridModelling
    using HybridModellingExperiments
    import HybridModellingExperiments: Model3SP, HybridFuncRespModel, LuxBackend, InferICs, run_simulations, LogMSELoss, save_results
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

    const model = HybridFuncRespModel()
    const rng = Random.MersenneTwister(1234)
    const callback(l, epoch, ts) = nothing

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
            verbose,
            rng,
            HlSize,
            kwargs...
    )
        bounds = NamedTuple([dp => cat(
                                 [sort([(1e0 - perturb / 2e0) * k,
                                      (1e0 + perturb / 2e0) * k])
                                  for k in p_true[dp]]...,
                                 dims = 2)' for dp in keys(p_true)])
        distrib_param = NamedTuple([dp => product_distribution([Uniform(
                                                       bounds[dp][i, 1], bounds[dp][i, 2])
                                                   for i in axes(bounds[dp], 1)])
                                    for dp in keys(p_true)])
        constraint = NamedTupleConstraint(NamedTuple([dp => BoxConstraint(
                                                          bounds[dp][:, 1], bounds[dp][
                                                              :, 2])
                                                      for dp in keys(p_true)]))
        p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

        parameters = ParameterLayer(; constraint, init_value = p_init)

        functional_response = Lux.Chain(Lux.Dense(2, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, HlSize, NNlib.tanh),
            Lux.Dense(HlSize, 2))
        lux_model = ODEModel(
            (; parameters, functional_response), model; alg, abstol, reltol, sensealg, maxiters, verbose)

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
    nruns = 5
    ic_estims = [
        InferICs(true,
            NamedTupleConstraint((;
                u0 = BoxConstraint([1e-3], [5e0])))),
        InferICs(false)]
    noises = [0.2, 0.4]
    weight_decays = [1e-9, 1e-5, 1e-1]
    perturbs = [1.0]
    lrs = [1e-3, 1e-2, 1e-1]
    batchsizes = [10]
    HlSizes = [2^2, 2^3, 2^4]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims,
        noise in noises, weight_decay in weight_decays, perturb in perturbs, lr in lrs,
        batchsize in batchsizes, HlSize in HlSizes

        optim_backend = LuxBackend(AdamW(; eta = lr, lambda = weight_decay),
            n_epochs,
            adtype,
            loss_fn,
            callback)

        data, p_true = generate_data(model; tspan, fixed_params...)

        varying_params = (; segmentsize,
            optim_backend,
            experimental_setup = infer_ic,
            noise,
            data,
            p_true,
            weight_decay,
            perturb,
            batchsize,
            HlSize)
        push!(pars_arr, varying_params)
    end
    return shuffle!(rng, pars_arr)
end

const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])
const adtype = AutoZygote()
const loss_fn = LogMSELoss()
const n_epochs = 2

mode = SerialMode()

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
    model
)

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)
save_results(string(@__FILE__); results)
