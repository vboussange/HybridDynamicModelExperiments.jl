#=
Second example showcasing the fit of a hybrid model,
where the neural net is overparameterizing the model.
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers
using SciMLSensitivity
include("../src/3sp_model.jl")
include("../src/hybrid_functional_response_model.jl")
include("../src/loss_fn.jl")

"""
    init(model::HybridFuncRespModel, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(model::HybridFuncRespModel2, perturb=0.5)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param_arr = Pair{Symbol, Any}[]

    for dp in keys(p_true)
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
abstol = 1e-6
reltol = 1e-6
tspan = (0., 800)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]
p_hybrid = ComponentArray(r = Float32[1.0, -0.4, -0.08])

hybrid_model = HybridFuncRespModel2(ModelParams(;p=p_hybrid,
                                        tspan,
                                        u0 = u0_true,
                                        alg,
                                        reltol,
                                        abstol,
                                        saveat = tsteps,
                                        verbose = false, # suppresses warnings for maxiters
                                        maxiters = 50_000,
                                        ))
sim = simulate(hybrid_model, 
                tspan=(tsteps[1], tsteps[end]), 
                u0=data[:, 1])
ax2 = Plots.plot(sim, title = "Initial prediction")
# feed_pred_gains(hybrid_model, u0_true, hybrid_model.mp.p)
Plots.plot(hcat([loss(hybrid_model, c, hybrid_model.mp.p) for c in eachcol(data)]...)', title="Feed pred gains")

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
Warr, Harr, qarr = create_sparse_matrices(model, model.mp.p)
feeding(model, data[:, 1], model.mp.p)
Plots.plot(hcat([feed_pred_gains(model, c, model.mp.p) for c in eachcol(data)]...)', title="Feed pred gains")

loss_likelihood = LossLikelihood()
p_bij, u0_bij = init(hybrid_model)
p_init = hybrid_model.mp.p
infprob = InferenceProblem(hybrid_model, p_init; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            p_bij, u0_bij)

# Inference
res_inf = inference(infprob;
                    data, 
                    group_size = 7, 
                    adtype=Optimization.AutoZygote(), 
                    epochs=[200, 50], 
                    tsteps = tsteps,
                    optimizers = [OptimizationOptimisers.Adam(1e-2), OptimizationOptimJL.LBFGS()],
                    verbose_loss = true,
                    info_per_its = 10,
                    multi_threading = false)

sim_res_inf = simulate(hybrid_model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
ax3 = Plots.plot(sim_res_inf, title="Predictions after training")
Plots.plot(ax, ax2, ax3)