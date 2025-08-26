#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5, BS3
import Turing: NUTS, HMC
import ADTypes: AutoZygote, AutoForwardDiff, AutoReverseDiff
using Plots
using Distributions
import Distributions: ProductNamedTupleDistribution
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridModelling
using HybridModellingExperiments
import HybridModellingExperiments: Model3SP, LogMSELoss, train, MCMCBackend, LuxBackend, InferICs, forecast, get_parameter_error
import Lux
using Random

"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(::Model3SP, ::MCMCBackend, p_true, perturb=1e0)
    parameter_priors = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])
    # Careful: float type is not easily imposed, see https://github.com/JuliaStats/Distributions.jl/issues/1995
    return (;parameters=parameter_priors)
end


# Model metaparameters
alg = BS3()
sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))
sampler = HMC(0.05, 4; adtype=AutoForwardDiff()) # fastest, by far
# sensealg = GaussAdjoint()
# adtype = AutoZygote()
# adtype = AutoReverseDiff(;compile=true) # fails
# adtype = AutoMooncake()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945]
p_true = (;H = [1.24, 2.5],
            q = [4.98, 0.8],
            r = [1.0, -0.4, -0.08],
            A = [1.0])
backend = MCMCBackend()
model = Model3SP()

# Lux model initialization with biased parameters
parameters = ParameterLayer(init_value = p_true) # p_true is only used for inferring length of parameters
lux_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, sensealg)
ps_priors = init(model, backend, p_true)

# True Lux model initialization
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

rng = MersenneTwister(1)

# Data generation
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
ax = Plots.scatter(tsteps, data', title = "Data")

# Defining inference problem
datadistrib = x -> LogNormal(log(max(x, 1e-6)))
# Model initialized with perturbed parameters
dataloader = SegmentedTimeSeries((data, tsteps), segmentsize=8, partial_batch = true)

## Testing Turing backend
# chain = train(MCMCBackend(),
#         InferICs(true);
#         datadistrib, 
#         model_priors = ps_priors,
#         model = lux_model, 
#         rng, 
#         dataloader, 
#         sampler = NUTS(; adtype), 
#         n_iterations = 1000)

## Testing Turing backend without ICs
res = train(MCMCBackend(),
            InferICs(false);
            datadistrib, 
            model_priors = ps_priors,
            model = lux_model, 
            rng, 
            dataloader, 
            sampler, 
            n_iterations = 1000,)

using StatsPlots
plot(res.chains)

tsteps_forecast = tspan[end]:4:tspan[end]+200
last_tok = tokens(tokenize(dataloader))[end]
segment_data, segment_tsteps = tokenize(dataloader)[last_tok]
forecasted_data = forecast(MCMCBackend(), res.st_model, res.chains, union(segment_tsteps, tsteps_forecast))
true_data = lux_true_model((;u0 = data[:, tsteps .∈ Ref(union(segment_tsteps, tsteps_forecast))][:, 1], tspan = (segment_tsteps[1], tsteps_forecast[end]), saveat = union(segment_tsteps, tsteps_forecast)), ps_true, st)[1]
ax = Plots.plot(union(segment_tsteps, tsteps_forecast), true_data', label = "true", title="Forecasted vs true data", linestyle = :dash, color = palette(:auto)[1:3]')
for pred in forecasted_data
    Plots.plot!(ax, union(segment_tsteps, tsteps_forecast), pred', label = "", color=Plots.palette(:auto)[1:3]', alpha=0.1)
end
ax

get_parameter_error(MCMCBackend(), res.st_model, res.chains, p_true)
