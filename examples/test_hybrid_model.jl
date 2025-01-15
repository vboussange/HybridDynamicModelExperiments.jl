#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
include("../src/3sp_model.jl")
include("../src/hybrid_functional_response_model.jl")
include("../src/loss_fn.jl")

# Model metaparameters
alg = Tsit5()
abstol = 1e-4
reltol = 1e-4
tspan = (0., 100)
tsteps = tspan[1]:1:tspan[2]
u0_true = Float32[0.5,0.8,0.5]
p_hybrid = ComponentArray(r = Float32[1.0, -0.4, -0.08],
                            A = Float32[1.0])

hybrid_model = HybridFuncRespModel(ModelParams(;p=p_hybrid,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))

plot(simulate(hybrid_model))

function loss(p)
    sum(simulate(hybrid_model; p) |> Array)
end

using DifferentiationInterface, Zygote
value_and_gradient(loss, AutoZygote(), hybrid_model.mp.p)