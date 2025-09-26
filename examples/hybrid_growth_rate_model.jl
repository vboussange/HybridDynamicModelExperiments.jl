#=
Short exampling showcasing the fit of a model with 
a neural network to account for functional response.
=#

cd(@__DIR__)
import OrdinaryDiffEqTsit5: Tsit5
import Turing: arraydist
import ADTypes: AutoZygote, AutoForwardDiff
using Plots
using Distributions
import Distributions: ProductNamedTupleDistribution
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridDynamicModels
import HybridDynamicModelExperiments: VaryingGrowthRateModel, HybridGrowthRateModel,
                                   LogMSELoss, train, SGDBackend, InferICs, forecast,
                                   get_parameter_error, growth_rate_resource,
                                   forecast, generate_noisy_data
import Lux
using Random
import NNlib

function init(
        model::HybridGrowthRateModel,
        ::SGDBackend;
        alg,
        abstol,
        reltol,
        sensealg,
        maxiters,
        p_true,
        perturb = 1e0,
        verbose,
        rng,
        growth_rate,
        kwargs...
)
    bounds = NamedTuple([dp => cat(
                             [sort([(1e0 - perturb / 2e0) * k,
                                  (1e0 + perturb / 2e0) * k])
                              for k in p_true[dp]]...,
                             dims = 2)' for dp in keys(p_true)])
    distrib_param = NamedTuple([dp => product_distribution([Uniform(
                                                                bounds[dp][i, 1], bounds[dp][
                                                                    i, 2])
                                                            for i in axes(bounds[dp], 1)])
                                for dp in keys(p_true)])
    constraint = NamedTupleConstraint(NamedTuple([dp => BoxConstraint(
                                                      bounds[dp][:, 1], bounds[dp][
                                                          :, 2])
                                                  for dp in keys(p_true)]))
    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    parameters = ParameterLayer(; constraint, init_value = p_init)

    lux_model = ODEModel(
        (; parameters, growth_rate), model; alg, abstol, reltol, sensealg, maxiters, verbose)

    return lux_model
end

rng = MersenneTwister(3)

# ODE solver
alg = Tsit5()
abstol = 1e-4
reltol = 1e-4
tspan = [0e0, 800e0]
tsteps = collect(550.0:4.0:800.0)

# Data generation
u0_true = [0.5, 0.8, 0.5]
p_true = (H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0],
    s = [1.0])
noise = 0.2
dudt_true = VaryingGrowthRateModel()
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel(
    (; parameters), dudt_true; alg, abstol, reltol, tspan, saveat = tsteps)

ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((; u0 = u0_true), ps_true, st)
# data_with_noise = rand(rng, arraydist(LogNormal.(log.(data), noise)))
data_with_noise = generate_noisy_data(data, noise, rng)
ax1 = Plots.scatter(tsteps, data_with_noise', title = "Data")

# Hybrid model metaparameters
sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))
adtype = AutoZygote()
HlSize = 2^3
growth_rate = Lux.Chain(Lux.Dense(1, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, 1))
segment_length = 4
batchsize = 10
loss_fn = LogMSELoss()
weight_decay = 1e-5
lr = 1e-2
epochs = 1000
optim_backend = SGDBackend(AdamW(eta = lr, lambda = weight_decay), epochs, adtype, loss_fn)
infer_ics = InferICs(true, NamedTupleConstraint((; u0 = BoxConstraint([1e-3], [5e0]))))
# model definition
truncated_p = (; H = p_true.H, q = p_true.q, r = p_true.r[2:end], A = p_true.A)
lux_model = init(HybridGrowthRateModel(),
    optim_backend;
    alg,
    abstol,
    reltol,
    sensealg,
    maxiters = 50_000,
    verbose = false,
    p_true = truncated_p,
    perturb = 1e0,
    growth_rate,
    rng)

# testing lux model
ftype = Lux.f64
ps, st = ftype(Lux.setup(rng, lux_model))

preds, _ = lux_model(
    (; u0 = [0.77, 0.060, 0.945] |> ftype,
        tspan = tspan |> ftype, saveat = tsteps |> ftype),
    ps,
    st)
ax2 = Plots.plot(tsteps, preds', title = "Initial preds.")
display(plot(ax1, ax2))

dataloader = SegmentedTimeSeries((data_with_noise, tsteps);
    segment_length,
    shift = segment_length - 2,
    batchsize,
    partial_batch = true) |> ftype

loss_fn(preds, preds)

## Testing Lux backend
res = train(optim_backend,
    lux_model,
    dataloader,
    infer_ics,
    rng, Lux.f64);

function plot_segments(dataloader, ps, st, ics)
    plt = plot()
    colors = [:blue, :red]
    dataloader = tokenize(dataloader)
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        _ics = ics[tok].u0
        pred = lux_model(
            (; u0 = _ics, saveat = segment_tsteps,
                tspan = (segment_tsteps[1], segment_tsteps[end])),
            ps,
            st)[1]
        color = colors[mod1(tok, 2)]
        # @show segment_data
        plot!(plt, segment_tsteps, segment_data',
            label = (tok == 1 ? "Data" : ""), color = color, linestyle = :solid)
        plot!(plt, segment_tsteps, pred', label = (tok == 1 ? "Predicted" : ""),
            color = color, linestyle = :dash)
    end
    return plt
end

plot_segments(dataloader, res.ps, st, res.ics)

function plot_forecast(
        tspan, dataloader, res, lux_true_model, data, tsteps, data_with_noise, ps_true, st)
    tsteps_forecast = tspan[end]:4:(tspan[end] + 200)
    last_tok = tokens(tokenize(dataloader))[end]
    segment_data, segment_tsteps = tokenize(dataloader)[last_tok]
    forecasted_data = forecast(
        optim_backend, lux_model, res.ps, res.st, res.ics, union(
            segment_tsteps, tsteps_forecast))
    true_data = lux_true_model(
        (; u0 = data[:, tsteps .∈ Ref(union(segment_tsteps, tsteps_forecast))][:, 1],
            tspan = (segment_tsteps[1], tsteps_forecast[end]),
            saveat = union(segment_tsteps, tsteps_forecast)),
        ps_true,
        st)[1]
    ax = Plots.plot(union(segment_tsteps, tsteps_forecast), forecasted_data',
        label = "forecasted", title = "Forecasted vs true data")
    Plots.plot!(ax, union(segment_tsteps, tsteps_forecast), true_data',
        label = "true", linestyle = :dash, color = palette(:auto)[1:3]')
    Plots.scatter!(ax, segment_tsteps, data_with_noise[:, tsteps .∈ Ref(segment_tsteps)]',
        label = "training data", color = palette(:auto)[1:3]', yscale = :log10)
    return ax
end

ax = plot_forecast(
    tspan, dataloader, res, lux_true_model, data, tsteps, data_with_noise, ps_true, st)
display(ax)

# plot growth rate
water_avail = collect(-1.0:0.05:1)'
rates, _ = growth_rate(water_avail, res.ps.growth_rate, st.growth_rate)
true_rates = growth_rate_resource.(Ref(p_true), water_avail)

ax = Plots.plot(water_avail[:], rates[:], label = "Neural network",
    title = "Growth rate", xlabel = "Water availability", ylabel = "Growth rate")
Plots.plot!(ax, water_avail[:], true_rates[:], label = "True")