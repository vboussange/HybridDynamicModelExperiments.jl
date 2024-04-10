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
include("../model/loss_fn.jl")
include("../model/utils.jl")

using SciMLSensitivity
using PiecewiseInference
using OptimizationFlux, OptimizationOptimJL
using JLD2

data_params = (datasize = 100,
                step = 10)

simul_params = (alg = Tsit5(),
                abstol = 1e-6,
                reltol = 1e-6,
                verbose=false, # suppresses warnings for maxiters
                maxiters=50_000,)

tsteps = range(500e0, step=data_params.step, length=data_params.datasize)
tspan = (0e0, tsteps[end])

p_true = ComponentArray(x_c = [0.4], 
                        x_p = [0.015], 
                        y_c = [2.01], 
                        y_p = [5.00], 
                        R_0 = [0.16129], 
                        C_0 = [0.5])
p_init = ComponentArray(x_c = [0.4 * (2. *rand())], 
                        x_p = [0.015 * (2. *rand())], 
                        y_c = [2.01 * (2. *rand())], 
                        y_p = [5.00 * (2. *rand())], 
                        R_0 = [0.16129 * (2. *rand())], 
                        C_0 = [0.5 * (2. *rand())])
u0_true = [0.5,0.8,0.5]

mp = ModelParams(; p=p_true,
    tspan,
    u0=u0_true,
    saveat=tsteps,
    simul_params...)

model_true = EcosystemModelMcCann(mp)

data = simulate(model_true, u0=u0_true) |> Array


model_params = (alg = Tsit5(),
                abstol = 1e-6,
                reltol = 1e-6,
                tspan = (tsteps[1], tsteps[end]),
                sensealg = ForwardDiffSensitivity(),
                # sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                saveat = tsteps,
                maxiters = 50_000,)

inference_params = (adtype = Optimization.AutoForwardDiff(),
                    group_size = 11,
                    data = data,
                    tsteps = tsteps,
                    optimizers = [Adam(1e-2)],
                    epochs = [2000],
                    verbose_loss = true,
                    info_per_its = 200,
                    multi_threading = false)

# plot_time_series(data)

u0_init = data[:,1]

mp = ModelParams(; p=p_init,
                u0=u0_init,
                model_params...)
model = EcosystemModelMcCann(mp)

using BenchmarkTools
@btime simulate(model)

simul_model = simulate(model) |> Array;

# plot_time_series(simul_model)

using Bijectors
u0_bij = bijector(Uniform(1f-3, 5f0)) # seems like stacked is not working with AD

# constraints on parameters
distrib_param = NamedTuple([dp => Product([Uniform(sort([5f-1 * k, 1.5f0 * k])...) for k in p_true[dp]]) for dp in keys(p_true)])

# bijectors, to constrain parameter values and initial conditions
p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])

loss_likelihood(data, pred, rg) = mean((data .- pred).^2)
loss_likelihood(data, pred) = loss_likelihood(data, pred, nothing)

infprob = InferenceProblem(model, p_true; 
                            loss_u0_prior = loss_likelihood, # loss applied on the initial conditions inferred
                            loss_likelihood = loss_likelihood, # loss applied on the simuation ouput
                            # p_bij, 
                            # u0_bij
                            );

@time res = inference(infprob;
                            inference_params...)

@btime PiecewiseInference.to_optim_space(p_init, infprob)

@btime PiecewiseInference.to_param_space(p_init, infprob)