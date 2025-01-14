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
include("../src/loss_fn.jl")

"""
    init(model::Model3SP, perturb=0.5)

Initialize parameters, parameter and initial condition constraints for the inference.
"""
function init(model::Model3SP, perturb=0.5)
    p_true = model.mp.p
    T = eltype(p_true)
    distrib_param = NamedTuple([dp => Product([Uniform(sort([(1f0-perturb/2f0) * k, (1f0+perturb/2f0) * k])...) for k in p_true[dp]]) for dp in keys(p_true)])


    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    u0_bij = bijector(Uniform(T(1e-3), T(5e0)))  # For initial conditions
    
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])
    p_init = ComponentArray(p_init) .|> eltype(data)

    return p_init, p_bij, u0_bij
end


# Model metaparameters
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0., 800)
tsteps = 550.:4.:800.
u0_true = Float32[0.5,0.8,0.5]
p_true = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        A = Float32[1.0])

# Model initialization with true parameters
model = Model3SP(ModelParams(;p= p_true,
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
fig = Plots.scatter(tsteps, data')

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LossLikelihood()
p_init, p_bij, u0_bij = initialize_constraints(model)
infprob = InferenceProblem(model, p_init; 
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
sim_res_inf = simulate(model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
Plots.plot!(fig, sim_res_inf)
