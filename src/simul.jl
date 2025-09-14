#=
Utilities to run inference simulations.
=#

using HybridModelling
using Distributions
using Bijectors
using Optimisers: Optimisers

function split_data(data, forecast_length)
    train_idx = 1:(size(data, 2) - forecast_length - 1)
    test_idx = (size(data, 2) - forecast_length):size(data, 2)
    return train_idx, test_idx
end

get_lr(opt::Optimisers.AbstractRule) = opt.eta

function init(
        model::AbstractEcosystemModel,
        ::LuxBackend;
        alg,
        abstol,
        reltol,
        sensealg,
        maxiters,
        p_true,
        perturb = 1e0,
        rng,
        verbose,
        kwargs...
)
    bounds = NamedTuple([dp => cat(
                             [sort([(1e0 - perturb / 2e0) * k, (1e0 + perturb / 2e0) * k])
                              for k in p_true[dp]]...,
                             dims = 2)' for dp in keys(p_true)])
    distrib_param = NamedTuple([dp => product_distribution([Uniform(bounds[dp][i, 1], bounds[dp][i, 2])
                                               for i in axes(bounds[dp], 1)])
                                for dp in keys(p_true)])
    constraint = NamedTupleConstraint(NamedTuple([dp => BoxConstraint(
                                                       bounds[dp][:, 1], bounds[dp][:, 2])
                                                   for dp in keys(p_true)]))
    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    parameters = ParameterLayer(; constraint, init_value = p_init)
    lux_model = ODEModel((; parameters), model; alg, abstol, reltol, sensealg, maxiters, verbose)

    return lux_model
end

function forecast(::LuxBackend, model, ps, st, ics, tsteps_forecast)
    u0 = ics[end].u0
    t0 = ics[end].t0
    return model(
        (;
            u0 = u0,
            saveat = tsteps_forecast,
            tspan = (t0, tsteps_forecast[end])
        ), ps, st)[1][
        :, :, 1
    ]
end

function get_parameter_error(::LuxBackend, model, ps, st, p_true)
    ps_tr, _ = model.components.parameters(ps.parameters, st.parameters)
    med_par_err = median([median(abs.((ps_tr[k] - p_true[k]) ./ p_true[k]))
                          for k in keys(ps_tr)])
    return med_par_err
end

# Simulation function, preprocess before train, train, and postprocess
# only valid for non hybrid models.
function simu(
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
        luxtype,
        kwargs...
)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()

    data_w_noise = generate_noisy_data(data, noise)
    train_idx, test_idx = split_data(data, forecast_length)
    dataloader = SegmentedTimeSeries(
        (data_w_noise[:, train_idx], tsteps[train_idx]);
        segmentsize,
        batchsize,
        shift,
        partial_batch = true
    )

    # Lux model initialization with biased parameters
    lux_model = init(model, optim_backend; p_true, sensealg, rng, kwargs...)
    println(
        "Launching simulations for segmentsize = $segmentsize, noise = $noise, backend = $(nameof(optim_backend)), experimental_setup = $(typeof(experimental_setup))",
    )

    med_par_err = missing
    forecast_err = missing
    ps = missing
    st = missing
    ics = missing

    try
        res = train(
            optim_backend, lux_model, dataloader, experimental_setup, rng, luxtype
        )

        ps = res.ps
        st = res.st
        ics = res.ics
        med_par_err = get_parameter_error(optim_backend, lux_model, ps, st, p_true)

        preds = forecast(optim_backend, lux_model, ps, st, ics, tsteps[test_idx])
        forecast_err = optim_backend.loss_fn(preds, data[:, test_idx])
    catch e
        println("Error occurred during training: ", e)
    end

    # saving states fails with JLD2
    return (;
        med_par_err,
        forecast_err,
        modelname = nameof(model),
        ps,
        ics,
        segmentsize,
        batchsize,
        shift,
        noise,
        lr = get_lr(optim_backend.opt),
        adtype = string(typeof(optim_backend.adtype)),
        sensealg = string(typeof(sensealg)),
        optim_backend = nameof(optim_backend),
        infer_ics = isestimated(experimental_setup),
        kwargs...
    )
end

function init(
        model::AbstractEcosystemModel,
        ::MCMCBackend;
        alg,
        abstol,
        reltol,
        sensealg,
        maxiters,
        p_true,
        perturb = 1.0f0,
        kwargs...
)
    parameter_priors = NamedTuple([dp => product_distribution([Uniform(sort([(1e0 - perturb / 2e0) * k,
                                                      (1e0 + perturb / 2e0) * k])...)
                                                  for
                                                  k in p_true[dp]]) for dp in keys(p_true)])
    # Careful: float type is not easily imposed, see https://github.com/JuliaStats/Distributions.jl/issues/1995
    parameters = BayesianLayer(ParameterLayer(), parameter_priors) # p_true is only used for inferring length of parameters
    lux_model = ODEModel((; parameters), model; alg, abstol, reltol, sensealg, maxiters)
    return lux_model
end

function forecast(::MCMCBackend, st_model, ics, chain, tsteps_forecast, nsamples = 100)
    nsamples = min(nsamples, size(chain, 1))
    last_tok = length(ics)
    last_ics_t0 = ics[last_tok].t0

    posterior_samples = sample(st_model, chain, nsamples; replace = false)
    preds = []
    for ps in posterior_samples
        pred = st_model(
            (;
                u0 = last_tok,
                saveat = tsteps_forecast,
                tspan = (last_ics_t0, tsteps_forecast[end])
            ),
            ps
        )
        push!(preds, pred)
    end
    return preds
end

function get_parameter_error(::MCMCBackend, st_model, chain, p_true, nsamples = 100)
    nsamples = min(nsamples, size(chain, 1))
    posterior_samples = sample(st_model, chain, nsamples; replace = false)
    err = []
    for ps in posterior_samples
        ps = ps.model.parameters
        med_par_err = median([median(abs.(ps[k] - p_true[k]) ./ p_true[k])
                              for k in keys(ps)])
        push!(err, med_par_err)
    end
    return median(err)
end

function simu(
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
        loss_fn = nothing,
        forecast_length = 10,
        rng,
        kwargs...
)

    # invoke garbage collection to avoid memory overshoot on SLURM
    GC.gc()

    data_w_noise = generate_noisy_data(data, noise)
    train_idx, test_idx = 1:(size(data, 2) - forecast_length - 1),
    (size(data, 2) - forecast_length):size(data, 2)
    dataloader = SegmentedTimeSeries(
        (data_w_noise[:, train_idx], tsteps[train_idx]); segmentsize, shift
    )

    # Lux model initialization with biased parameters
    lux_model = init(model, optim_backend; p_true, sensealg, kwargs...)
    println(
        "Launching simulations for segmentsize = $segmentsize, noise = $noise, backend = $(nameof(optim_backend)), experimental_setup = $(typeof(experimental_setup))",
    )

    med_par_err = missing
    forecast_err = missing
    try
        res = train(optim_backend, lux_model, dataloader, experimental_setup, rng)

        med_par_err = get_parameter_error(optim_backend, res.st_model, res.chains, p_true)

        if !isnothing(loss_fn)
            preds = forecast(
                optim_backend, res.st_model, res.ics, res.chains, tsteps[test_idx])
            forecast_err = median(loss_fn(p, data[:, test_idx]) for p in preds)
        end
    catch e
        println("Error occurred during training: $e")
    end

    return (;
        med_par_err,
        forecast_err,
        model = nameof(model),
        segmentsize,
        noise,
        sampler = string(typeof(optim_backend.sampler)),
        optim_backend = nameof(optim_backend),
        sensealg = string(typeof(sensealg)),
        infer_ics = isestimated(experimental_setup),
        kwargs...
    )
end