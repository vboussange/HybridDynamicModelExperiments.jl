#=
Short exampling showcasing the fit of a 3 species model.
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
import HybridDynamicModelExperiments: Model7SP, LogMSELoss, train, SGDBackend, InferICs, forecast, get_parameter_error
import Lux
using Random

const FloatType = Float64

"""
    init(model::Model7SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(::Model7SP, ::SGDBackend, p_true, perturb=1e0)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    u0_transform = bijector(Uniform(1e-3, 5e0))  # For initial conditions
    
    # TODO: problem with rand(Uniform), casts to Float64
    p_init = NamedTuple([k => rand(distrib_param[k])  .|> FloatType for k in keys(distrib_param)])

    return p_init, p_transform, u0_transform
end

# Model metaparameters
alg = Tsit5()
sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
adtype = AutoZygote()
# sensealg = GaussAdjoint()
# adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945, 0.467, 0.18, 0.14, 0.18]
batchsize = 10
p_true = (ω = [0.2],
            H = [2.89855, 7.35294, 8.0, 2.89855, 7.35294, 12.0],
            q = [1.38, 0.272, 1e-1 ,1.38, 0.272, 5e-2],
            r = [1.0, -0.15, -0.08, 1.0, -0.15, -0.01, -0.005],
            A = [1.0, 1.0])
model = Model7SP()
p_init, p_transform, u0_transform = init(model, SGDBackend(), p_true)

# Lux model initialization with biased parameters
parameters = ParameterLayer(constraint = Constraint(p_transform), 
                            init_value = p_init)
lux_model = ODEModel((;parameters), Model7SP(); alg, abstol, reltol, sensealg)

# True Lux model initialization
parameters = ParameterLayer(constraint = NoConstraint(), 
                            init_value = p_true)
lux_true_model = ODEModel((;parameters), Model7SP(); alg, abstol, reltol, tspan, saveat = tsteps)

rng = MersenneTwister(1)

# Data generation
σ = 0.1
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
data_with_noise = rand(arraydist(LogNormal.(log.(data), σ)))
ax = Plots.scatter(tsteps, data_with_noise', title = "Data")

# Defining inference problem
# Model initialized with perturbed parameters
segment_length = 8
dataloader = SegmentedTimeSeries((data_with_noise, tsteps); 
                                segment_length, 
                                shift=segment_length-2, 
                                batchsize,
                                partial_batch = true)

## Testing Lux backend
res = train(SGDBackend(),
            InferICs(true);
            model = lux_model, 
            rng, 
            dataloader, 
            opt = Adam(1e-3), 
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
forecasted_data = forecast(SGDBackend(), res.best_model, union(segment_tsteps, tsteps_forecast))
true_data = lux_true_model((;u0 = data[:, tsteps .∈ Ref(union(segment_tsteps, tsteps_forecast))][:, 1], tspan = (segment_tsteps[1], tsteps_forecast[end]), saveat = union(segment_tsteps, tsteps_forecast)), ps_true, st)[1]
ax = Plots.plot(union(segment_tsteps, tsteps_forecast), forecasted_data', label = "forecasted", title="Forecasted vs true data")
Plots.plot!(ax, union(segment_tsteps, tsteps_forecast), true_data', label = "true", linestyle = :dash, color = palette(:auto)[1:3]')
Plots.scatter!(ax, segment_tsteps, data_with_noise[:, tsteps .∈ Ref(segment_tsteps)]', label = "training data", color = palette(:auto)[1:3]')

get_parameter_error(SGDBackend(), res.best_model, p_true)

# @code_warntype train(SGDBackend(),
#                     InferICs(true);
#                     model = lux_model, 
#                     rng, 
#                     dataloader, 
#                     opt = Adam(1e-2), 
#                     adtype,
#                     n_epochs = 1000)

