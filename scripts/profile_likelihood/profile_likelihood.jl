#=
Calculating profile likelihood for figure 1, based on `SimpleEcosystemModel3SP`
=#

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
include("../../src/3sp_model.jl")
include("../../src/loss_fn.jl")
include("../../src/utils.jl")
include("../../src/loss_fn.jl")
include("../../src/plotting.jl")

using SciMLSensitivity
using PiecewiseInference
using OptimizationFlux, OptimizationOptimJL
using JLD2

noise = 2f-1
data_params = (datasize = 100,
                step = 4)

tsteps = range(500f0, step=data_params.step, length=data_params.datasize)
tspan = (0f0, tsteps[end])

model_params = (alg = Tsit5(),
                abstol = 1e-6,
                reltol = 1e-6,
                tspan = (tsteps[1], tsteps[end]),
                sensealg = ForwardDiffSensitivity(),
                saveat = tsteps,
                maxiters = 50_000,)


u0_true = Float32[0.77, 0.060, 0.945]
p_true = ComponentArray(H = Float32[1.24, 2.5],
                    q = Float32[4.98, 0.8],
                    r = Float32[1.0, -0.4, -0.08],
                    K₁₁ = Float32[1.0],
                    A = Float32[1.0])

mp = ModelParams(;tspan,
                u0=u0_true,
                saveat=tsteps,
                model_params...)

model = SimpleEcosystemModel3SP(mp)

data = simulate(model, u0=u0_true, p=p_true) |> Array
data_set_w_noise = data .* exp.(noise * randn(size(data)))
loss = LossLikelihood()
ranges = get_ranges(;group_size=11, datasize = data_params.datasize)
u0_bij = identity
p_bij = (r = identity, b = identity)
u0s_init = data[:,first.(ranges),:][:]
infprob = InferenceProblem(model, p_true, loss_u0_prior = (x, y) -> 0, loss_likelihood = loss )
idx_ranges = 1:length(ranges) # idx of batches

# naive_loss
function naive_loss(p)
    sol = simulate(model; p) |> Array
    return loss(data_set_w_noise, sol), sol
end

# normal loss
function piecewise_loss(p)
    θ = PiecewiseInference._build_θ(p, u0s_init, infprob)
    PiecewiseInference.piecewise_loss(infprob,
                                        θ,
                                        data_set_w_noise,
                                        tsteps,
                                        ranges,
                                        idx_ranges,
                                        false)
end

params = - (0.04:0.0003:0.12)
p_likelihood = []
p_likelihood_piecewise = []

for param in params
    p = copy(p_true)
    p.r[3] = param
    push!(p_likelihood, naive_loss(p)[1])
    push!(p_likelihood_piecewise, piecewise_loss(p)[1])
end

results = Dict("p_likelihood" => p_likelihood, 
                "p_likelihood_piecewise" => p_likelihood_piecewise,
                "params" => params,
                "p_true" => p_true.r[3])
save(split(@__FILE__,".")[1]*".jld2", results)


# if plotting
fig, ax = subplots()
ax.plot(collect(params), p_likelihood_piecewise)
ax.plot(collect(params), p_likelihood)
ax.set_yscale("log")
fig