#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import Distributions: ProductNamedTuple
import OrdinaryDiffEq: Tsit5
import Turing: HMC
import ADTypes: AutoZygote
using Plots
using Distributions
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridModelling
import HybridModellingBenchmark: Model3SP, LogMSELoss, train, TuringBackend, LuxBackend, InferICs
import Lux
using Random

const FloatType = Float64

"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(::Model3SP, ::LuxBackend, p_true, perturb=1f0)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

    p_transform = Bijectors.NamedTransform(NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)]))
    u0_transform = bijector(Uniform(1f-3, 5f0))  # For initial conditions
    
    # TODO: problem with rand(Uniform), casts to Float64
    p_init = NamedTuple([k => rand(distrib_param[k])  .|> FloatType for k in keys(distrib_param)])

    return p_init, p_transform, u0_transform
end

function init(::Model3SP, ::TuringBackend, p_true, perturb=1f0)
    distrib_param = ProductNamedTupleDistribution(NamedTuple([dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)]))
    # Careful: see https://github.com/JuliaStats/Distributions.jl/issues/1995
    p_init = Lux.fmap(FloatType, rand(distrib_param))

    return p_init, p_prior
end


# Model metaparameters
alg = Tsit5()
sensealg = GaussAdjoint()
adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tspan = (0f0, 800f0)
tsteps = 550f0:4f0:800f0
u0_true = Float32[0.77, 0.060, 0.945]
p_true = (;H = Float32[1.24, 2.5],
            q = Float32[4.98, 0.8],
            r = Float32[1.0, -0.4, -0.08],
            A = Float32[1.0])

model = Model3SP()
p_init, p_transform, u0_transform = init(model, p_true)

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
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((;u0 = u0_true), ps_true, st)
ax = Plots.scatter(tsteps, data', title = "Data")

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LogMSELoss()
dataloader = SegmentedTimeSeries((data, tsteps), segmentsize=8, partial_batch = true)

## Testing Lux backend
# train(LuxBackend(),
#     InferICs(false);
#     model = lux_model, 
#     rng, 
#     dataloader, 
#     opt = Adam(5e-3), 
#     adtype,
#     n_epochs = 1000)

## Testing Turing backend
train(TuringBackend(),
    InferICs(true);
    model = lux_model, 
    rng, 
    dataloader, 
    sampler = HMC(0.05, 4; adtype), 
    n_iterations = 1000, 
    σ=0.1)