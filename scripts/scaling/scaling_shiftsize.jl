#=
Short exampling showcasing the fit of a 3 species model. Showcases the use of a scheduler.
=#
cd(@__DIR__)
import OrdinaryDiffEqTsit5: Tsit5
import Turing: arraydist
import ADTypes: AutoZygote, AutoForwardDiff
using Plots
using Distributions
import Distributions: product_distribution
using Bijectors
using Optimisers
using SciMLSensitivity
using HybridDynamicModels
import HybridDynamicModelExperiments: Model3SP, LogMSELoss, train, SGDBackend, InferICs,
                                   forecast, get_parameter_error, save_results
import Lux
using Random
using BenchmarkTools
using DataFrames

function init_parameters(rng, p_true, perturb=1e0)
    bounds = NamedTuple([dp => cat([sort([(1e0-perturb/2e0) * k, (1e0+perturb/2e0) * k]) for k in p_true[dp]]..., dims=2)' for dp in keys(p_true)])
    distrib_param = NamedTuple([dp => Product([Uniform(bounds[dp][i, 1], bounds[dp][i, 2]) for i in axes(bounds[dp], 1)]) for dp in keys(p_true)])

    constraints = NamedTupleConstraint(NamedTuple([dp => BoxConstraint(bounds[dp][:, 1], bounds[dp][:, 2]) for dp in keys(p_true)]))
    
    p_init = NamedTuple([k => rand(rng, distrib_param[k]) for k in keys(distrib_param)])

    return p_init, constraints
end

rng = MersenneTwister(2)

# Model metaparameters
alg = Tsit5()
sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))
adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tspan = (0e0, 800e0)
tsteps = 550e0:4e0:800e0
u0_true = [0.77, 0.060, 0.945]
p_true = (; H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0])

lr_init = 1e-2
callback(l, epoch, ts) = nothing

loss_fn = LogMSELoss()
dudt = Model3SP()
p_init, constraint = init_parameters(rng, p_true)
u0_constraint = NamedTupleConstraint((;u0 = BoxConstraint([1e-3], [5e0]))) # For initial conditions

# Lux model initialization with biased parameters
parameters = ParameterLayer(;constraint,
    init_value = p_init)
lux_model = ODEModel((; parameters), dudt; alg, abstol, reltol, sensealg)

# True Lux model initialization
parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel((; parameters), dudt; alg, abstol, reltol, tspan, saveat = tsteps)

# Data generation
σ = 0.4
ps_true, st = Lux.setup(rng, lux_true_model)
data, _ = lux_true_model((; u0 = u0_true), ps_true, st)

segmentsize = 8
batchsize = 10
shifts = [2, 4, 6, 8]
results = []
for shift in shifts, infer_ics in (true, false)
    @info "Benchmarking shift: $shift, infer_ics: $infer_ics"
    dataloader = SegmentedTimeSeries((data, tsteps);
        segmentsize,
        partial_batch = true,
        batchsize,
        shift,
    )
    optim_backend = SGDBackend(Adam(lr_init), 1, adtype, loss_fn, callback)
    stats = @benchmark train($optim_backend,
        lux_model,
        $dataloader,
        $(InferICs(infer_ics)),
        rng
    ) setup=(rng = MersenneTwister(2))
    push!(results, (; times = stats.times, batchsize, memory = stats.memory, allocs = stats.allocs, infer_ics))
end

df = DataFrame(results)
save_results(@__FILE__; results = df)
