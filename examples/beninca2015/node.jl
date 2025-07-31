#=
in Model hybrid 2, all parameters are functions of time
=#
cd(@__DIR__)
import OrdinaryDiffEq: Tsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using SciMLSensitivity
using Dates, DataFrames, CSV
using LinearAlgebra

include("../../src/beninca/beninca_model.jl")
include("../../src/beninca/node.jl")
include("../../src/loss_fn.jl")

data_df = load_data()
train_idx, test_idx = 1:50, 51:size(data_df, 1)
data = Matrix(data_df[:, 2:end])'
data_training = data[:, train_idx]
tsteps = data_df[:, 1]
tsteps_days = Dates.date2epochdays.(tsteps) .|> Float64
tsteps_days_training = tsteps_days[train_idx]
tsteps_days_test = tsteps_days[test_idx]
# NOTE: to convert back to Date, 
# tsteps_reconstructed = Dates.epochdays2date.(round.(Int, tsteps_days))


Plots.plot(tsteps, data')

ax_data = Plots.plot(tsteps[train_idx], data_training', 
                    title = "Data", 
                    labels = reshape(names(data_df)[2:end], 1, :), 
                    ylim = (0, 1))

# Model metaparameters
alg = Tsit5()
abstol = 1e-3
reltol = 1e-3
tspan_training = (tsteps_days_training[1], tsteps_days_training[end]) # 10 years in days

# B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
# overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
# mussels)
u0_init = [0.103, 0.019, 0.033, 0.040] 

# Model initialization with true parameters
model_hybrid = node(ModelParams(;
                                tspan = tspan_training,
                                u0 = u0_init,
                                alg,
                                sensealg= BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                                reltol,
                                abstol,
                                saveat = tsteps_days_training,
                                verbose = true,
                                maxiters = 50_000,
                                ); seed=123)

# Data generation
# TODO: we need to fine-tune the neural net and parameter bijectors to obtain stable simulations
# But first check if the model is well coded by plugging in a simple mortality function
# NOTE: it could be that unstable simulations are the cause of the `inference` failing
sim0 = simulate(model_hybrid, u0 = data[:,5]) |> Array # works
ax_sim0 = Plots.plot(tsteps[train_idx], sim0', title = "Initial guess", labels = ["B₀" "Bᵢ" "A" "M"])
Plots.plot(ax_sim0, ax_data, layout= @layout([a; b]), size = (800, 600))


u0_bij = bijector(Uniform(1e-6, 1.0))

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LossLikelihood()
loss_param_prior(p) = norm(p.p_nn) * 1e0 # regularization on neural network parameters
infprob = InferenceProblem(model_hybrid, model_hybrid.mp.p; 
                            u0_bij,
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            # loss_param_prior = loss_param_prior
                            )

# debugging
# using Zygote
# function loss_fn(p)
#     return sum(simulate(model_hybrid, p=p, u0 = data[:,5], tspan=(tsteps_days[1], tsteps_days[10]), saveat = tsteps_days[1:10]))
# end
# loss_fn(model_hybrid.mp.p) # works
# Zygote.gradient(loss_fn, model_hybrid.mp.p) # works, but reltol, abstol should be low

# Inference
res_inf = inference(infprob;
                    data=Matrix(data_training), 
                    group_size = 6, # months
                    adtype=Optimization.AutoZygote(),
                    batchsizes = [1, 1],
                    epochs=[2000, 200], 
                    tsteps = tsteps_days_training,
                    optimizers = [OptimizationOptimisers.Adam(1e-2), 
                                OptimizationOptimisers.Adam(1e-4)],
                    verbose_loss = true,
                    info_per_its = 100,
                    multi_threading = false) # Minimum loss for all batches: 3027.13037109375

# Simulation with inferred parameters
sim_res_inf = simulate(model_hybrid, p = res_inf.p_trained, u0=data[:, 2])
ax_full_pred = Plots.plot(sim_res_inf, title="Full length prediction after training", ylim=(0, 1))

data_pred = hcat([
    Matrix(
        simulate(
            model_hybrid,
            p = res_inf.p_trained,
            tspan = (tsteps_days_training[first(res_inf.ranges[i])], tsteps_days_training[last(res_inf.ranges[i])]),
            saveat = tsteps_days_training[res_inf.ranges[i]],
            u0 = res_inf.u0s_trained[i]
        )[:, 2:end]
    ) 
    for i in 1:length(res_inf.ranges)
]...)

ax_data_pred = Plots.plot(
    tsteps_days_training[2:end], data_pred', 
    title = "Predicted (piecewise)", 
    labels = ["B₀" "Bᵢ" "A" "M"],
    # ylim = (0, 1)
)

myplot = Plots.plot(ax_data, ax_data_pred, ax_full_pred; layout = @layout([a; b; c]), size = ( 800, 900))
display(myplot)

p_true = ComponentArray(
    c_BR = [0.018],
    c_AB = [0.049],
    c_M = [0.078],
    c_AR = [0.021],
    m_B = [0.003],
    m_A = [0.011],
    m_M = [0.017],
    α = [0.1],
)

println("Inferred parameters:")
for k in keys(res_inf.p_trained)
    if k == :p_nn
        continue
    end
    println("$(k): inferred = $(res_inf.p_trained[k]), original = $(p_true[k])")
end


# Plotting predictions based on the last u0s trained

# idx_start = first(res_inf.ranges[end])
# idx_end = test_idx[20]
# data_extrapolated = simulate(
#             model_hybrid,
#             p = res_inf.p_trained,
#             tspan = (tsteps_days[idx_start], tsteps_days[idx_end]),
#             saveat = 10.,
#             u0 = res_inf.u0s_trained[end]
#         )[:, 2:end]
# plot(data_extrapolated)
# scatter!(tsteps_days[idx_start:idx_end], data[:, idx_start:idx_end]',
#     ylim = (0, 1),
#     color = Plots.palette(:auto)[1:4]')