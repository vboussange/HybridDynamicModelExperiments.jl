#=
Generates a time series from simulating `Model3SP`, further used to create figure 1.
=#

cd(@__DIR__)
using Graphs
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
using OptimizationOptimisers, OptimizationOptimJL
using JLD2

using PiecewiseInference
import PiecewiseInference: AbstractODEModel

noise = 2f-1
data_params = (datasize = 20,
                step = 8)

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

model = Model3SP(mp)

data = simulate(model, u0=u0_true, p=p_true) |> Array
data_set_w_noise = data .* exp.(noise * randn(size(data)))
loss = LossLikelihood()
ranges = get_ranges(;group_size=6, datasize = data_params.datasize)
idx_ranges = 1:length(ranges) # idx of batches

u0s_init = data[:,first.(ranges),:][:]

@save "data_set_w_noise.jld2" data_set_w_noise tsteps ranges