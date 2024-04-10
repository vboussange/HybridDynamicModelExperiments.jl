cd(@__DIR__)
using Graphs
using EcoEvoModelZoo
using ParametricModels
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using SparseArrays
using ComponentArrays
using PythonPlot
include("3sp_model.jl")
include("../model/loss_fn.jl")
include("../model/utils.jl")

using SciMLSensitivity
using PiecewiseInference
using OptimizationFlux, OptimizationOptimJL
using JLD2

@load "3sp_model_simul.jld2" data tsteps species_colors node_labels p_true

model_params = (alg = Tsit5(),
                abstol = 1e-6,
                reltol = 1e-6,
                tspan = (tsteps[1], tsteps[end]),
                sensealg = ForwardDiffSensitivity(),
                # sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                saveat = tsteps,
                maxiters = 50_000,)

inference_params = (adtype = Optimization.AutoForwardDiff(),
                    # adtype = Optimization.AutoForwardDiff(),
                    group_size = 11,
                    batchsizes = [10],
                    data = data,
                    tsteps = tsteps,
                    optimizers = [ADAM(1e-2)],
                    epochs = [100],
                    verbose_loss = true,
                    info_per_its = 1,
                    multi_threading = false)

plot_time_series(data)

u0_init = data[:,1]

using Bijectors
u0_bij = bijector(Uniform(1f-3, 5f0)) # seems like stacked is not working with AD

# constraints on parameters
distrib_param = NamedTuple([dp => Product([Uniform(sort([5f-1 * k, 1.5f0 * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

# bijectors, to constrain parameter values and initial conditions
p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])

# initialising initial guess
p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])
p_init = ComponentArray(p_init) .|> eltype(data)


mp = ModelParams(; p=p_init,
                u0=u0_init,
                model_params...)
model = SimpleEcosystemModel3SP(mp)

@time simul_model = simulate(model) |> Array;

plot_time_series(simul_model)

# baseline inference
infprob = InferenceProblem(model, p_init; 
                            loss_u0_prior = loss_likelihood, # loss applied on the initial conditions inferred
                            loss_likelihood = loss_likelihood, # loss applied on the simuation ouput
                            p_bij, u0_bij);

@time res = inference(infprob; inference_params...)
#=
 24.856091 seconds (454.71 M allocations: 45.766 GiB, 19.59% gc time)
 =#

# no p_bij
infprob = InferenceProblem(model, p_init; 
                            loss_u0_prior = loss_likelihood, # loss applied on the initial conditions inferred
                            loss_likelihood = loss_likelihood);
#=
 77.228275 seconds (595.39 M allocations: 58.204 GiB, 7.19% gc time, 62.67% compilation time)
 =#

@time res = inference(infprob; inference_params...)

# reverse mode
mp = ModelParams(; p=p_init,
                u0=u0_init,
                model_params..., 
                sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),)
model = SimpleEcosystemModel3SP(mp)

infprob = InferenceProblem(model, p_init; 
                            loss_u0_prior = loss_likelihood, # loss applied on the initial conditions inferred
                            loss_likelihood = loss_likelihood, # loss applied on the simuation ouput
                            p_bij, u0_bij);

@time res = inference(infprob; inference_params..., adtype = Optimization.AutoZygote())
#=
  5.692957 seconds (28.91 M allocations: 2.179 GiB, 4.36% gc time)
=#