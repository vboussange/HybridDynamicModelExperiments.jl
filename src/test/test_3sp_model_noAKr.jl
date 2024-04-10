using OrdinaryDiffEq
using UnPack
using SparseArrays
using Graphs
using BenchmarkTools
include("../plotting.jl")
include("../3sp_model_noA.jl")

alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0., 800)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]


p_3SP = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K = Float32[1])

p_Omniv3SP = ComponentArray(H = Float32[1.24, 6.25,2.5],
                        q = Float32[4.98, 0.32,0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K = Float32[1],
                        ω = Float32[0.])

model3SP = SimpleEcosystemModel3SP(ModelParams(;p = p_3SP,
                                                tspan,
                                                u0 = u0_true,
                                                alg,
                                                reltol,
                                                abstol,
                                                saveat = tsteps,
                                                verbose = false, # suppresses warnings for maxiters
                                                maxiters = 50_000,
                                                ))

@btime simulate($model3SP); # 5.228 ms (150889 allocations: 9.90 MiB)
@assert eltype(simulate(model3SP)) <: Float32
plot_time_series(simulate(model3SP), model3SP)

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

@btime simulate($modelOmniv3SP) # 6.315 ms (181003 allocations: 10.67 MiB)
@assert eltype(simulate(modelOmniv3SP)) <: Float32
plot_time_series(simulate(modelOmniv3SP), modelOmniv3SP)

# Non zero ω
p_Omniv3SP = ComponentArray(H = [1.24, 6.25,2.5],
                        q = [4.98, 0.32,0.8],
                        r = [1.0, -0.4, -0.08],
                        K = [1.0],
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
plot_time_series(simulate(modelOmniv3SP), modelOmniv3SP)