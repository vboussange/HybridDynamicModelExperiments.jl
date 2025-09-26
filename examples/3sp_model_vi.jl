#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEqTsit5: Tsit5
import ADTypes: AutoZygote, AutoForwardDiff
using Plots
using Distributions
using Optimisers
using SciMLSensitivity
using HybridDynamicModels
import HybridDynamicModelExperiments: Model3SP, LogMSELoss, train, MCSamplingBackend, VIBackend, InferICs, forecast
import Lux
using Random


"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(::Model3SP, ::Union{MCSamplingBackend, VIBackend}, p_true, perturb=1e0)
    parameter_priors = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])
    # Careful: float type is not easily imposed, see https://github.com/JuliaStats/Distributions.jl/issues/1995
    return (;parameters=parameter_priors)
end


# Model metaparameters
alg = Tsit5()
sensealg = ForwardDiffSensitivity()
adtype = AutoForwardDiff()
# sensealg = GaussAdjoint()
# adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945]
p_true = (;H = [1.24, 2.5],
            q = [4.98, 0.8],
            r = [1.0, -0.4, -0.08],
            A = [1.0])
backend = VIBackend()
model = Model3SP()

# Lux model initialization with biased parameters
parameters = ParameterLayer()
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
# chain = train(MCSamplingBackend(),
#         InferICs(true);
#         datadistrib, 
#         model_priors = ps_priors,
#         model = lux_model, 
#         rng, 
#         dataloader, 
#         sampler = NUTS(; adtype), 
#         n_iterations = 1000)

# TODO: sometimes, errors with log provided with - value.
## Testing Turing backend without ICs
res = train(backend,
        InferICs(true);
        datadistrib, 
        model_priors = ps_priors,
        model = lux_model, 
        rng,
        dataloader, 
        n_iterations = 1000, 
        adtype
        )
z = rand(res.q_avg, 1000)
