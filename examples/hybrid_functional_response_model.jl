#=
Short exampling showcasing the fit of a model with 
a neural network to account for functional response.
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

"""
    init(model::HybridFuncRespModel, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(model::HybridFuncRespModel, perturb=0.5)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param_arr = Pair{Symbol, Any}[]

    for dp in [:r, :A]
        dp == :p_nn && continue
        pair = dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]])
        push!(distrib_param_arr, pair)
    end
    pair_nn = :p_nn => Uniform(-Inf, Inf)
    push!(distrib_param_arr, pair_nn)

    distrib_param = NamedTuple(distrib_param_arr)

    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions
    
    return p_bij, u0_bij
end

# Model metaparameters
alg = Tsit5()
abstol = 1e-4
reltol = 1e-4
tspan = (0., 800)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]
p_hybrid = ComponentArray(r = Float32[0.5, -0.2, -0.1],
                            A = Float32[1.15])

hybrid_model = HybridFuncRespModel(ModelParams(;p=p_hybrid,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        sensealg= BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))
sim = simulate(hybrid_model, 
                tspan=(tsteps[1], tsteps[end]), 
                u0=u0_true)
ax2 = Plots.plot(sim, title = "Initial prediction")
feeding(hybrid_model, u0_true, hybrid_model.mp.p)

p_full = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        A = Float32[1.0])

model = Model3SP(ModelParams(;p=p_full,
                            tspan,
                            u0 = u0_true,
                            alg,
                            reltol,
                            abstol,
                            saveat = tsteps,
                            verbose = false, # suppresses warnings for maxiters
                            maxiters = 50_000,
                            ))

data = simulate(model) |> Array # 19.720 ms
ax = Plots.scatter(tsteps, data', title = "Data")
feeding(model, data[:, 1], model.mp.p)

loss_likelihood = LossLikelihood()
p_bij, u0_bij = init(hybrid_model)
p_init = hybrid_model.mp.p
infprob = InferenceProblem(hybrid_model, p_init; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            p_bij, u0_bij)

# Inference
@time res_inf = inference(infprob;
                            data, 
                            group_size = 5, 
                            adtype=Optimization.AutoZygote(), 
                            epochs=[500, 50], 
                            tsteps = tsteps,
                            optimizers = [OptimizationOptimisers.Adam(2e-2), OptimizationOptimJL.LBFGS()],
                            verbose_loss = true,
                            info_per_its = 10,
                            multi_threading = false)
# Simulation with inferred parameters
# feed_pred_gains(hybrid_model, data[:, 1], res_inf.p_trained)

sim_res_inf = simulate(hybrid_model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
ax3 = Plots.plot(sim_res_inf, title="Predictions after training")
Plots.plot(ax, ax2, ax3)


true_feeding_rates = hcat([feeding(model, c,p_full).nzval for c in eachcol(data)]...)
inferred_feeding_rates = hcat([feeding(hybrid_model, c, res_inf.p_trained).nzval for c in eachcol(data)]...)

ax4 = Plots.scatter(data[1:2,:]', true_feeding_rates', title="True feeding rates", ylim=(0., 5.))
ax5 = Plots.scatter(data[1:2,:]', inferred_feeding_rates', title="Inferred feeding rates", ylim=(0., 5.))
Plots.plot(ax4, ax5)

println("True r: ", p_full.r)
println("Inferred r: ", res_inf.p_trained.r)
println("True A: ", p_full.A)
println("Inferred A: ", res_inf.p_trained.A)

ax4 = Plots.plot(hcat([feed_pred_gains(hybrid_model, c, res_inf.p_trained) for c in eachcol(data)]...)', title="Inferred feed pred gains", ylims=(-1., 1))
ax5 = Plots.plot(hcat([feed_pred_gains(model, c, p_full) for c in eachcol(data)]...)', title="True feed pred gains", ylims=(-1., 1))
Plots.plot(ax4, ax5)