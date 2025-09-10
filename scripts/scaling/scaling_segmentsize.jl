import Distributed: @everywhere
import HybridModellingExperiments: setup_distributed_environment
setup_distributed_environment(4)

@everywhere begin 
    using Lux
    using HybridModellingExperiments
    using HybridModelling
    import HybridModellingExperiments: Model3SP, LuxBackend, MCMCBackend, InferICs,
                                    run_simulations, LogMSELoss, save_results
    import HybridModellingExperiments: SerialMode, ParallelMode, DistributedMode
    import OrdinaryDiffEqTsit5: Tsit5
    import SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
    import ADTypes: AutoZygote, AutoForwardDiff
    import Optimisers: Adam
    import Turing: HMC

    using Dates
    using Random
    using JLD2
    using DataFrames
    using Distributions
    using Dates
    using BenchmarkTools

    function HybridModellingExperiments.simu(
            optim_backend::LuxBackend,
            experimental_setup::InferICs;
            model,
            p_true,
            segmentsize,
            batchsize,
            shift = nothing,
            noise,
            data,
            tsteps,
            sensealg,
            forecast_length = 10,
            rng,
            kwargs...
    )

        data_w_noise = HybridModellingExperiments.generate_noisy_data(data, noise)
        train_idx, test_idx = HybridModellingExperiments.split_data(data, forecast_length)
        dataloader = SegmentedTimeSeries(
            (data_w_noise[:, train_idx], tsteps[train_idx]);
            segmentsize,
            batchsize,
            shift,
            partial_batch = true
        )

        # Lux model initialization with biased parameters
        lux_model = HybridModellingExperiments.init(model, optim_backend; p_true, sensealg, rng, kwargs...)
        println(
            "Benchmarking segmentsize = $segmentsize, noise = $noise, backend = $(HybridModellingExperiments.nameof(optim_backend)), experimental_setup = $(typeof(experimental_setup))",
        )

        time = missing
        memory = missing
        allocs = missing
        try
            stats = eval(:(@benchmark train(
                $optim_backend, $lux_model, $dataloader, $experimental_setup, $rng
            )))
            time = stats.times
            allocs = stats.allocs
            memory = stats.memory
        catch e
            println("Error occurred during training: ", e)
        end

        return (;
            modelname = HybridModellingExperiments.nameof(model),
            time,
            memory,
            allocs,
            segmentsize,
            sensealg = string(typeof(sensealg)),
            optim_backend = HybridModellingExperiments.nameof(optim_backend),
            infer_ics = HybridModellingExperiments.istrue(experimental_setup),
        )
    end


    function HybridModellingExperiments.simu(
            optim_backend::MCMCBackend,
            experimental_setup;
            model,
            p_true,
            segmentsize,
            shift = nothing,
            noise,
            data,
            tsteps,
            sensealg,
            forecast_length = 10,
            rng,
            kwargs...
    )

        data_w_noise = HybridModellingExperiments.generate_noisy_data(data, noise)
        train_idx, test_idx = HybridModellingExperiments.split_data(data, forecast_length)
        dataloader = SegmentedTimeSeries(
            (data_w_noise[:, train_idx], tsteps[train_idx]);
            segmentsize,
            shift,
            partial_batch = true
        )

        # Lux model initialization with biased parameters
        lux_model = HybridModellingExperiments.init(model, optim_backend; p_true, sensealg, kwargs...)
        println(
            "Benchmarking segmentsize = $segmentsize, noise = $noise, backend = $(HybridModellingExperiments.nameof(optim_backend)), experimental_setup = $(typeof(experimental_setup))",
        )

        time = missing
        try
            stats = eval(:(@benchmark train($optim_backend, $lux_model, $dataloader, $experimental_setup, $rng)))
            time = stats.times

        catch e
            println("Error occurred during training: $e")
        end

        return (;
            modelname = HybridModellingExperiments.nameof(model),
            time,
            segmentsize,
            sensealg = string(typeof(sensealg)),
            optim_backend = HybridModellingExperiments.nameof(optim_backend),
            infer_ics = HybridModellingExperiments.istrue(experimental_setup),
        )
    end
end

function generate_data(model::Model3SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
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
    segmentsizes = floor.(Int, exp.(range(log(2), log(100), length = 6)))
    models = [Model3SP(), Model5SP(), Model7SP()]
    ic_estims = [InferICs(true), InferICs(false)]
    datadistrib = x -> LogNormal(log(max(x, 1e-6)))

    backends = [
        LuxBackend(
            Adam(1e-2), nits, AutoZygote(), loss_fn; verbose_frequency = Inf),
        MCMCBackend(
            HMC(0.05, 4, adtype = AutoForwardDiff()), nits, datadistrib; progress = false)]

    pars_arr = []
    for segmentsize in segmentsizes, infer_ic in ic_estims, model in models, optim_backend in backends

        data, p_true = generate_data(model; tspan, fixed_params...)
        varying_params = (; segmentsize,
            optim_backend,
            experimental_setup = infer_ic,
            model,
            data,
            p_true)
        push!(pars_arr, varying_params)
    end
    return shuffle!(fixed_params.rng, pars_arr)
end

mode = DistributedMode()

const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])
const nits = 1 # number of epochs or iterations depending on the context
loss_fn = LogMSELoss()

fixed_params = (alg = Tsit5(),
    abstol = 1e-4,
    reltol = 1e-4,
    tsteps,
    maxiters = 50_000,
    sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
    rng = Random.MersenneTwister(1234),
    batchsize = 10,
    forecast_length = 10, # not used but we keep it for consistency
    noise = 0.2)


simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")

println("Launching simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)