#=
Short exampling showcasing the fit of a model with 
a neural network to account for functional response.
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

function initialize_constraints(model::HybridFuncRespModel)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param_arr = Pair{Symbol, Any}[]

    for dp in keys(p_true)
        dp == :p_nn && continue
        pair = dp => Product([Uniform(sort(T[0.25 * k, 1.75 * k])...) for k in p_true[dp]])
        push!(distrib_param_arr, pair)
    end
    pair_nn = :p_nn => Uniform(-Inf, Inf)
    push!(distrib_param_arr, pair_nn)


    distrib_param = NamedTuple(distrib_param_arr)
    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions

    return p_bij, u0_bij
end

alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0., 800)
tsteps = 550.:4.:800.
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

feed_pred_gains(hybrid_model, u0_true, hybrid_model.mp.p)
res = simulate(hybrid_model) # 19.720 ms
Plots.plot(res)

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
Warr, Harr, qarr = create_sparse_matrices(model, model.mp.p)
feed_pred_gains(model, data[:, 1], model.mp.p)
res = simulate(model) # 19.720 ms
Plots.plot(res)
data = res |> Array

loss_likelihood = LossLikelihood()
p_bij, u0_bij = initialize_constraints(hybrid_model)

infprob = InferenceProblem(hybrid_model, hybrid_model.mp.p; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            p_bij, u0_bij)

res_inf = inference(infprob;
                        data, 
                        group_size = 7, 
                        adtype=Optimization.AutoZygote(), 
                        epochs=[200], 
                        tsteps = tsteps,
                        optimizers = [OptimizationOptimisers.Adam(5e-2)],
                        verbose_loss = true,
                        info_per_its = 10,
                        multi_threading = false)
feed_pred_gains(hybrid_model, data[:, 1], res_inf.p_trained)
sim_res_inf = simulate(hybrid_model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
plot(sim_res_inf)
Plots.plot(ax, ax2, ax3)