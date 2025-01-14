#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers
using SciMLSensitivity
include("../src/3sp_model.jl")
include("../src/hybrid_growth_rate_model.jl")
include("../src/loss_fn.jl")

"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(model::HybridGrowthRateModel, perturb=0.5)
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
abstol = 1e-4
reltol = 1e-4
tspan = (0., 800)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]
p_true = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        A = Float32[1.0],
                        s = Float32[1.])

# Model initialization with true parameters
model = Model3SPVaryingGrowthRate(ModelParams(;p= p_true,
                                                tspan,
                                                u0 = u0_true,
                                                alg,
                                                reltol,
                                                abstol,
                                                saveat = tsteps,
                                                verbose = false, # suppresses warnings for maxiters
                                                maxiters = 50_000,
                                                ))

# Data generation
data = simulate(model) |> Array # 19.720 ms
ax = Plots.scatter(tsteps, data', title = "Data")

# Model initialized with perturbed parameters
# Defining inference problem
p_hybrid = ComponentArray(H = Float32[1., 2.2],
                        q = Float32[4.3, 1.],
                        r = Float32[-0.4, -0.08],
                        A = Float32[1.0])

hybrid_model = HybridGrowthRateModel(ModelParams(;p=p_hybrid,
                                                u0 = u0_true, #TODO: required to get dim of model, but this should be modified
                                                alg,
                                                reltol,
                                                abstol,
                                                verbose = false, # suppresses warnings for maxiters
                                                maxiters = 50_000,
                                                ))
ax2 = Plots.plot(simulate(hybrid_model, 
                    tspan=(tsteps[1], tsteps[end]), 
                    u0=data[:, 1]), title = "Initial prediction")

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
                    epochs=[200], 
                    tsteps = tsteps,
                    optimizers = [OptimizationOptimisers.Adam(5e-2)],
                    verbose_loss = true,
                    info_per_its = 10,
                    multi_threading = false)
# Simulation with inferred parameters
# intinsic_growth_rate(hybrid_model, hybrid_model.mp.p, 0.)
# intinsic_growth_rate(hybrid_model, res_inf.p_trained, 0.)
# intinsic_growth_rate(model, model.mp.p, 0.)


sim_res_inf = simulate(hybrid_model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
ax3 = Plots.plot(sim_res_inf, title="Predictions after training")
Plots.plot(ax, ax2, ax3)
