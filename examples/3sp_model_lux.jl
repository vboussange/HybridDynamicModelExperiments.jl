#=
Short exampling showcasing the fit of a 3 species model. Showcases the use of a scheduler.
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
using HybridModelling
import HybridModellingExperiments: Model3SP, LogMSELoss, train, LuxBackend, InferICs, forecast, get_parameter_error
import Lux
using Random
import Flux
using ParameterSchedulers

function init_parameters(rng, p_true, perturb=1e0)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    
    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    return p_init, p_transform
end

rng = MersenneTwister(2)

# Model metaparameters
alg = Tsit5()
# sensealg = ForwardDiffSensitivity()
# adtype = AutoForwardDiff()
sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
# sensealg = GaussAdjoint()
adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945]
p_true = (;H = [1.24, 2.5],
            q = [4.98, 0.8],
            r = [1.0, -0.4, -0.08],
            A = [1.0])


lr_init = 1e-2

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

backend = LuxBackend(Adam(lr_init), 1000, adtype, loss_fn, callback)
dudt = Model3SP()
p_init, p_transform = init_parameters(rng, p_true)
u0_constraint = Constraint(Bijectors.NamedTransform((;u0 = bijector(Uniform(1e-3, 5e0)))))  # For initial conditions


# Lux model initialization with biased parameters
parameters = ParameterLayer(constraint = Constraint(p_transform), 
                            init_value = p_init)
lux_model = ODEModel((;parameters), dudt; alg, abstol, reltol, sensealg)

# True Lux model initialization
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel((;parameters), dudt; alg, abstol, reltol, tspan, saveat = tsteps)


# Data generation
σ = 0.1
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
data_with_noise = rand(rng, arraydist(LogNormal.(log.(data), σ)))
ax = Plots.scatter(tsteps, data_with_noise', title = "Data")

# Defining inference problem
# Model initialized with perturbed parameters
segmentsize = 8
dataloader = SegmentedTimeSeries((data_with_noise, tsteps); 
                                segmentsize, 
                                shift=segmentsize-2, 
                                partial_batch = true,
                                batchsize = 10
                                )

## Testing Lux backend
res = train(backend,
            lux_model,
            dataloader,
            InferICs(true),
            rng,
            # u0_constraint
            )

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

get_parameter_error(backend, lux_model, res.ps, res.st, p_true)
loss_fn(forecasted_data, true_data)
# @code_warntype train(LuxBackend(),
#                     InferICs(true);
#                     model = lux_model, 
#                     rng, 
#                     dataloader, 
#                     opt = Adam(1e-2), 
#                     adtype,
#                     n_epochs = 1000)