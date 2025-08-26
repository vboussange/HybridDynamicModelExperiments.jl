using OrdinaryDiffEq
using UnPack
using SparseArrays
using Graphs
using BenchmarkTools
include("../plotting.jl")
include("../3sp_model.jl")

alg = BS3()
abstol = 1e-3
reltol = 1e-3
tspan = (0., 800.)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]


p_3SP = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K₁₁ = Float32[1.0],
                        A = Float32[1.0])

p_Omniv3SP = ComponentArray(H = Float32[1.24, 6.25,2.5],
                        q = Float32[4.98, 0.32,0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K₁₁ = Float32[1.0],
                        A = Float32[1.0],
                        ω = Float32[0.])

model3SP = Model3SP(ModelParams(;p = p_3SP,
                                                tspan,
                                                u0 = u0_true,
                                                alg,
                                                reltol,
                                                abstol,
                                                saveat = tsteps,
                                                verbose = false, # suppresses warnings for maxiters
                                                maxiters = 50_000,
                                                ))

@btime simulate($model3SP); # 19.720 ms
@assert eltype(simulate(model3SP)) <: Float32

modelOmniv3SP = SimpleEcosystemModelOmniv3SP(ModelParams(;p = p_Omniv3SP,
                                                tspan,
                                                u0 = u0_true,
                                                alg,
                                                reltol,
                                                abstol,
                                                saveat = tsteps,
                                                verbose = false, # suppresses warnings for maxiters
                                                maxiters = 50_000,
                                                ))

@btime simulate($modelOmniv3SP) #2.791
@assert eltype(simulate(modelOmniv3SP)) <: Float32

# Non zero ω

p_Omniv3SP = ComponentArray(H = [1.24, 6.25,2.5],
                        q = [4.98, 0.32,0.8],
                        r = [1.0, -0.4, -0.08],
                        K₁₁ = [1.0],
                        A = [1.0],
                        ω = [0.4])

modelOmniv3SP = SimpleEcosystemModelOmniv3SP(ModelParams(;p = p_Omniv3SP,
                        tspan,
                        u0 = u0_true,
                        alg,
                        reltol,
                        abstol,
                        saveat = tsteps,
                        verbose = false, # suppresses warnings for maxiters
                        maxiters = 50_000,
                        ))

@btime simulate($modelOmniv3SP) #2.791

plot_time_series(simulate(modelOmniv3SP))