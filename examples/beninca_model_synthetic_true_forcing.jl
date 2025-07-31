#=
Synthetic fit with true forcing
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers
using SciMLSensitivity
using Dates, DataFrames, CSV
using Interpolations

include("../src/beninca_model.jl")
include("../src/loss_fn.jl")

# Model metaparameters
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0f0, 50 * 365f0) # 50 years in days
tsteps = (30f0*365:30f0:tspan[2]) .|> Float32 # every 30 days, starting from 30 years

# B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
# overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
# mussels)
u0_true = Float32[0.8, 0.1, 0.05, 0.1] 
p_true = ComponentArray(
    c_BR = [0.018f0],
    c_AB = [0.049f0],
    c_M = [0.078f0],
    c_AR = [0.021f0],
    m_B = [0.003f0],
    m_A = [0.011f0],
    m_M = [0.017f0],
    α = [0.1f0],
)

forcing = TrueTempForcing()

# Model initialization with true parameters
model = BenincaModel(ModelParams(;p= p_true,
                            tspan,
                            u0 = u0_true,
                            alg,
                            reltol,
                            abstol,
                            saveat = tsteps,
                            verbose = false, # suppresses warnings for maxiters
                            maxiters = 50_000,
                            ), forcing)

# Data generation
data = simulate(model) |> Array # 19.720 ms
ax_data = Plots.plot(tsteps, data', title = "Data", labels = ["B₀" "Bᵢ" "A" "M"])

"""
    init(model::BenincaModel, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(model::BenincaModel, perturb=1f0)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])


    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions
    
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])
    p_init = ComponentArray(p_init) .|> eltype(data)

    return p_init, p_bij, u0_bij
end


# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LossLikelihood()
p_init, p_bij, u0_bij = init(model)
infprob = InferenceProblem(model, p_init; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            p_bij, u0_bij)
ax_sim0 = Plots.plot(simulate(model, p=p_init), title="Initial guess")
# Inference
res_inf = inference(infprob;
                    data, 
                    group_size = 7, 
                    adtype=Optimization.AutoZygote(), 
                    epochs=[400], 
                    tsteps = tsteps,
                    optimizers = [OptimizationOptimisers.Adam(5e-2)],
                    verbose_loss = true,
                    info_per_its = 10,
                    multi_threading = false)

# Simulation with inferred parameters
sim_res_inf = simulate(model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
ax_pred = Plots.plot(sim_res_inf, title="Predictions after training", )
Plots.plot(ax_data, ax_sim0, ax_pred)