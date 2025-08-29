#=
Short exampling showcasing the fit of a model with 
a neural network to account for functional response.
=#

cd(@__DIR__)
import OrdinaryDiffEqTsit5, BS3
import Turing: arraydist
import ADTypes: AutoZygote, AutoForwardDiff
using Plots
using Distributions
import Distributions: ProductNamedTupleDistribution
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridModelling
import HybridModellingExperiments: VaryingGrowthRateModel, HybridGrowthRateModel, LogMSELoss, train, LuxBackend, InferICs, forecast, get_parameter_error
import Lux
using Random
import NNlib

rng = MersenneTwister(1)

function init(::LuxBackend, p_true, perturb=1e0)
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
u0_true = [0.5,0.8,0.5]
p_true = (H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [1.0])
σ = 0.1
dudt_true = VaryingGrowthRateModel()
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel((;parameters), dudt_true; alg, abstol, reltol, tspan, saveat = tsteps)

ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
data_with_noise = rand(arraydist(LogNormal.(log.(data), σ)))
ax = Plots.scatter(tsteps, data_with_noise', title = "Data")

# Hybrid model metaparameters
p_init = (H = [1., 2.2],
        q = [4.3, 1.],
        r = [-0.4, -0.08],
        A = [1.0])
sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
adtype = AutoZygote()
HlSize = 5
segmentsize = 9
batchsize = 10

# model definition
p_transform = Bijectors.NamedTransform((r = bijector(Uniform(-1.0, 1.0)),
                                        A = bijector(Uniform(0.0, 2.0))))
u0_transform = bijector(Uniform(1e-3, 5e0))
parameters = ParameterLayer(constraint = Constraint(p_transform), 
                            init_value = p_init)
growth_rate =  Lux.Chain(Lux.Dense(1, HlSize, NNlib.tanh),
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, HlSize, NNlib.tanh), 
                                Lux.Dense(HlSize, 1, NNlib.tanh))

dudt = HybridGrowthRateModel()
lux_model = ODEModel((;parameters, growth_rate), dudt; alg, abstol, reltol, sensealg)

# testing lux model
ps, st = Lux.setup(rng, lux_model)
ps = ps |> Lux.f64
preds, _ = lux_model((;u0 = [0.77, 0.060, 0.945], tspan, saveat = tsteps), ps, st)
Plots.plot(tsteps, preds', title="Initial predictions from hybrid model")


dataloader = SegmentedTimeSeries((data_with_noise, tsteps); 
                            segmentsize, 
                            shift=segmentsize-2, 
                            batchsize,
                            partial_batch = true)

## Testing Lux backend
res = train(LuxBackend(),
            InferICs(false);
            model = lux_model, 
            rng, 
            dataloader, 
            opt = Adam(1e-2), 
            adtype,
            n_epochs = 1000)

function plot_segments(dataloader, st_model)
    plt = plot()
    colors = [:blue, :red]
    for (batched_tokens, (batched_segments, batched_tsteps)) in tokenize(dataloader)

        batched_pred = st_model((batched_tokens, batched_tsteps))
        for (tok, segment_tsteps, segment_data, pred) in zip(batched_tokens, 
                                                            eachslice(batched_tsteps, dims=ndims(batched_tsteps)), 
                                                            eachslice(batched_segments, dims=ndims(batched_segments)), 
                                                            eachslice(batched_pred, dims=ndims(batched_pred)))
            color = colors[mod1(tok, 2)]
            plot!(plt, segment_tsteps, segment_data', label=(tok == 1 ? "Data" : ""), color=color, linestyle=:solid)
            plot!(plt, segment_tsteps, pred', label=(tok == 1 ? "Predicted" : ""), color=color, linestyle=:dash)
        end
    end

    display(plt)
    return plt
end

plot_segments(dataloader, res.best_model)


tsteps_forecast = tspan[end]:4:tspan[end]+200
last_tok = tokens(tokenize(dataloader))[end]
segment_data, segment_tsteps = tokenize(dataloader)[last_tok]
forecasted_data = forecast(LuxBackend(), res.best_model, union(segment_tsteps, tsteps_forecast))
true_data = lux_true_model((;u0 = data[:, tsteps .∈ Ref(union(segment_tsteps, tsteps_forecast))][:, 1], tspan = (segment_tsteps[1], tsteps_forecast[end]), saveat = union(segment_tsteps, tsteps_forecast)), ps_true, st)[1]
ax = Plots.plot(union(segment_tsteps, tsteps_forecast), forecasted_data', label = "forecasted", title="Forecasted vs true data")
Plots.plot!(ax, union(segment_tsteps, tsteps_forecast), true_data', label = "true", linestyle = :dash, color = palette(:auto)[1:3]')
Plots.scatter!(ax, segment_tsteps, data_with_noise[:, tsteps .∈ Ref(segment_tsteps)]', label = "training data", color = palette(:auto)[1:3]')
