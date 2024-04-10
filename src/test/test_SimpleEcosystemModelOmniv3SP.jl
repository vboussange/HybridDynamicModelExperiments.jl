#=
Testing `SimpleEcosystemModelOmniv3SP` used in actual simulations,
against `EcosystemModelOmnivory` implemented in EcoEvoModelZoo.jl

- 20X difference 
- 2 X difference between custom model omniv and 
=#

using EcoEvoModelZoo
using ParametricModels
using Random; Random.seed!(1234)
using LinearAlgebra
using UnPack
using Test
using OrdinaryDiffEq
using Statistics
using ComponentArrays
using Graphs
false ? (using PyPlot) : nothing
using BenchmarkTools
# Set a random seed for reproduceable behaviour
include("../plotting.jl")
include("../3sp_model.jl")
include("../3sp_model_cust.jl")


# Model from EcoEvoModelZoo
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0., 100)
tsteps = 1:1:100
p_ev = (x_c = [0.4], 
        x_p = [0.08], 
        y_c = [2.01], 
        y_pr = [2.00], 
        y_pc = [5.0], 
        R_0 = [0.16129], 
        R_02 = [ 0.5], 
        C_0 = [0.5],
        ω =[0.1])

u0_true = [0.5,0.8,0.5]


model = EcosystemModelOmnivory(ModelParams(;p = p_ev,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))

p_act = ComponentArray(H = [1.24, 6.25,2.5],
                        q = [4.98, 0.32,0.8],
                        r = [1.0, -0.4, -0.08],
                        K₁₁ = [1.0],
                        A = [1.0],
                        ω = [0.1])

# Piecewise inference model
actual_model = SimpleEcosystemModelOmniv3SP(ModelParams(;p = p_act,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))
# custom model
cust_model = SimpleEcosystemModelOmniv3SPCust(ModelParams(;p = p_act,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))

@btime simulate($model); # 56.833 μs (884 allocations: 70.91 KiB)
ideal_sol = simulate(model, u0 = u0_true)
display(plot_time_series(ideal_sol, actual_model))


@btime simulate($actual_model); # 1.003 ms (45434 allocations: 3.07 MiB)
actual_sol = simulate(actual_model, u0 = u0_true)
display(plot_time_series(actual_sol, actual_model))

@btime simulate($cust_model); # 1.003 ms (45434 allocations: 3.07 MiB)
cust_sol = simulate(actual_model, u0 = u0_true)
display(plot_time_series(actual_sol, actual_model))

@assert isapprox(actual_sol |> Array, ideal_sol |> Array, rtol=5e-2) |> all
@assert isapprox(actual_sol |> Array, cust_sol |> Array, rtol=5e-2) |> all
