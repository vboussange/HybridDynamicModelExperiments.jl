#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5, BS3
import Turing: arraydist
import ADTypes: AutoZygote, AutoForwardDiff
using Plots
using Distributions
import Distributions: ProductNamedTupleDistribution
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridModelling
import HybridModellingBenchmark: Model3SP, LogMSELoss, train, MCMCBackend, LuxBackend, InferICs
import Lux
using Random

const FloatType = Float64

"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(::Model3SP, ::LuxBackend, p_true, perturb=1e0)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    u0_transform = bijector(Uniform(1e-3, 5e0))  # For initial conditions
    
    # TODO: problem with rand(Uniform), casts to Float64
    p_init = NamedTuple([k => rand(distrib_param[k])  .|> FloatType for k in keys(distrib_param)])

    return p_init, p_transform, u0_transform
end

# Model metaparameters
alg = Tsit5()
sensealg = ForwardDiffSensitivity()
adtype = AutoForwardDiff()
# sensealg = GaussAdjoint()
# adtype = AutoZygote()
abstol = 1e-3
reltol = 1e-3
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945]
p_true = (;H = FloatType[1.24, 2.5],
            q = FloatType[4.98, 0.8],
            r = FloatType[1.0, -0.4, -0.08],
            A = FloatType[1.0])
model = Model3SP()
p_init, p_transform, u0_transform = init(model, LuxBackend(), p_true)

# Lux model initialization with biased parameters
parameters = ParameterLayer(constraint = Constraint(p_transform), 
                            init_value = p_init)
lux_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, sensealg)

# True Lux model initialization
parameters = ParameterLayer(constraint = NoConstraint(), 
                            init_value = p_true)
lux_true_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

rng = MersenneTwister(1)

# Data generation
σ = 0.1
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
data = rand(arraydist(LogNormal.(log.(data), σ)))
ax = Plots.scatter(tsteps, data', title = "Data")

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LogMSELoss()
dataloader = SegmentedTimeSeries((data, tsteps), segmentsize=8, partial_batch = true)

## Testing Lux backend
res = train(LuxBackend(),
            InferICs(true);
            model = lux_model, 
            rng, 
            dataloader, 
            opt = Adam(1e-2), 
            adtype,
            n_epochs = 1000)

# @code_warntype train(LuxBackend(),
#                     InferICs(true);
#                     model = lux_model, 
#                     rng, 
#                     dataloader, 
#                     opt = Adam(1e-2), 
#                     adtype,
#                     n_epochs = 1000)