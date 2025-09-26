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
import HybridDynamicModelExperiments: Model3SP, HybridFuncRespModel, LogMSELoss, train, SGDBackend, InferICs, forecast, get_parameter_error
import Lux
using Random
import NNlib

function init(::SGDBackend, p_true, perturb=1e0)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    u0_transform = bijector(Uniform(1e-3, 5e0))  # For initial conditions
    
    # TODO: problem with rand(Uniform), casts to Float64
    p_init = NamedTuple([k => rand(distrib_param[k])  .|> FloatType for k in keys(distrib_param)])

    return p_init, p_transform, u0_transform
end

# ODE solver
alg = Tsit5()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550.:4.:800.


# Data generation
u0_true = [0.77, 0.060, 0.945]
p_true = (;H = [1.24, 2.5],
            q = [4.98, 0.8],
            r = [1.0, -0.4, -0.08],
            A = [1.0])
σ = 0.4
model = Model3SP()

# True Lux model initialization
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel((;parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
data_with_noise = rand(arraydist(LogNormal.(log.(data), σ)))
Plots.scatter(tsteps, data_with_noise', label = "true data", title="Synthetic data with noise")

# training parameters
loss_fn = LogMSELoss()
segmentsize = 9
batchsize = 10
lr_init = 1e-2
weight_decay = 1e-9
dataloader = SegmentedTimeSeries((data_with_noise, tsteps); 
                            segmentsize, 
                            shift=segmentsize-2, 
                            batchsize,
                            partial_batch = true)

function callback(l, epoch, ts)
    if epoch % 10 == 0
        @info "Epoch $epoch: Loss = $l"
    end
end

# s = Step(lr_init, 0.8, 300)
# function callback(l, epoch, ts)
#     if epoch % 10 == 0
#         @info "Epoch $epoch: Loss = $l"
#     end
#     lr = s(epoch)
#     Optimisers.adjust!(ts.optimizer_state, lr)
# end
opt = AdamW(; eta = lr_init, lambda = weight_decay)
# opt = Adam(lr_init)
backend = SGDBackend(opt, 1000, adtype, loss_fn, callback)

# Hybrid model metaparameters
p_init = (r = [0.5, -0.2, -0.1],
          A = [1.15])
sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
adtype = AutoZygote()
HlSize = 2^3

rng = MersenneTwister(1)


# model definition
p_transform = Bijectors.NamedTransform((r = bijector(Uniform(-1.0, 1.0)),
                                        A = bijector(Uniform(0.0, 2.0))))
u0_transform = bijector(Uniform(1e-3, 5e0))
parameters = ParameterLayer(constraint = Constraint(p_transform), 
                            init_value = p_init)
functional_response =  Lux.Chain(Lux.Dense(2, HlSize, NNlib.tanh),
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, 2))

dudt = HybridFuncRespModel()
lux_model = ODEModel((;parameters, functional_response), dudt; alg, abstol, reltol, sensealg)

# testing lux model
ps, st = Lux.setup(rng, lux_model)
ps = ps |> Lux.f64
preds, _ = lux_model((;u0 = [0.77, 0.060, 0.945], tspan, saveat = tsteps), ps, st)
Plots.plot(tsteps, preds', title="Initial predictions from hybrid model")


## Testing Lux backend
res = train(backend,
            lux_model, 
            dataloader, 
            InferICs(true),
            rng)

function plot_segments(dataloader, res)
    plt = plot()
    colors = [:blue, :red]
    dataloader = tokenize(dataloader)
    for tok in tokens(dataloader)
        segment_data, segment_tsteps = dataloader[tok]
        _ics = res.ics[tok].u0
        pred = lux_model((;u0 = _ics, saveat = segment_tsteps, tspan = (segment_tsteps[1], segment_tsteps[end])), res.ps, res.st)[1]
        color = colors[mod1(tok, 2)]
        # @show segment_data
        plot!(plt, segment_tsteps, segment_data', label=(tok == 1 ? "Data" : ""), color=color, linestyle=:solid)
        plot!(plt, segment_tsteps, pred', label=(tok == 1 ? "Predicted" : ""), color=color, linestyle=:dash)
    end

    return plt
end

plt = plot_segments(dataloader, res)
display(plt)

tsteps_forecast = tspan[end]:4:tspan[end]+200
last_tok = tokens(tokenize(dataloader))[end]
segment_data, segment_tsteps = tokenize(dataloader)[last_tok]
forecasted_data = forecast(backend, lux_model, res.ps, res.st, res.ics, union(segment_tsteps, tsteps_forecast))
true_data = lux_true_model((;u0 = data[:, tsteps .∈ Ref(union(segment_tsteps, tsteps_forecast))][:, 1], tspan = (segment_tsteps[1], tsteps_forecast[end]), saveat = union(segment_tsteps, tsteps_forecast)), ps_true, st)[1]
ax = Plots.plot(union(segment_tsteps, tsteps_forecast), forecasted_data', label = "forecasted", title="Forecasted vs true data")
Plots.plot!(ax, union(segment_tsteps, tsteps_forecast), true_data', label = "true", linestyle = :dash, color = palette(:auto)[1:3]')
Plots.scatter!(ax, segment_tsteps, data_with_noise[:, tsteps .∈ Ref(segment_tsteps)]', label = "training data", color = palette(:auto)[1:3]')
# display(ax)
get_parameter_error(backend, lux_model, res.ps, res.st, p_true)
loss_fn(forecasted_data, true_data)