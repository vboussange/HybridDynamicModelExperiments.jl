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
using Dates, DataFrames, CSV
using LinearAlgebra

include("../../src/beninca/beninca_model.jl")
include("../../src/loss_fn.jl")

MyType = MyType
data_df = load_data(MyType)
train_idx, test_idx = 1:200, 201:size(data_df, 1)
data = Matrix(data_df[:, 2:end])'
data_training = data[:, train_idx]
tsteps = data_df[:, 1]
tsteps_days = Dates.date2epochdays.(tsteps) .|> MyType
tsteps_days_training = tsteps_days[train_idx]
tsteps_days_test = tsteps_days[test_idx]
# NOTE: to convert back to Date, 
# tsteps_reconstructed = Dates.epochdays2date.(round.(Int, tsteps_days))

ax_data = Plots.plot(tsteps, 
                    data', 
                    title = "Data", 
                    labels = reshape(names(data_df)[2:end], 1, :), 
                    ylim = (0, 1))

# constructing the interpolation for the forcing data
# forcing = DummyForcing(0.0)
# forcing = TrueTempForcing(MyType) # 1D true forcing
# forcing = TrueForcing(MyType, ["Temperature [oC]"], 10)
# forcing = TrueForcing(MyType, ["Wind run [km/day]"], 20)
# forcing = TrueForcing(MyType, ["Wave height [m]"], 30)
forcing = TrueForcing(MyType)

ax_forcing = Plots.plot(tsteps, hcat(forcing.(tsteps_days)...)', title = "Forcing")
Plots.plot(ax_data, ax_forcing, layout = @layout([a; b]), size = (800, 400))

# Model metaparameters
alg = Tsit5()
abstol = 1e-3
reltol = 1e-3
tspan_training = (tsteps_days_training[1], tsteps_days_training[end]) # 10 years in days

# B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
# overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
# mussels)
u0_init = [0.103, 0.019, 0.033, 0.040] 
p_init = ComponentArray(
                c_BR = [0.018],
                c_AB = [0.0888],
                c_M = [0.04],
                c_AR = [0.0019],
                m_B = [0.0002])

# Model initialization with true parameters
model_hybrid = hybrid_beninca_model(ModelParams(;p= p_init,
                            tspan = tspan_training,
                            u0 = u0_init,
                            alg,
                            sensealg= BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)),
                            reltol,
                            abstol,
                            saveat = tsteps_days_training,
                            verbose = true,
                            maxiters = 50_000,
                            ), forcing; seed=123)

# Data generation
# TODO: we need to fine-tune the neural net and parameter bijectors to obtain stable simulations
# But first check if the model is well coded by plugging in a simple mortality function
# NOTE: it could be that unstable simulations are the cause of the `inference` failing
sim0 = simulate(model_hybrid, u0 = data[:,5]) |> Array # works
ax_sim0 = Plots.plot(tsteps[train_idx], sim0', title = "Initial guess", labels = ["B₀" "Bᵢ" "A" "M"])
Plots.plot(ax_sim0, ax_data)

p_bij = (c_BR = bijector(Uniform(1e-4, 0.5)),
          c_AB = bijector(Uniform(1e-4, 0.5)),
          c_M = bijector(Uniform(1e-4, 0.5)),
          c_AR = bijector(Uniform(1e-4, 0.5)),
          m_B = bijector(Uniform(1e-4, 0.5)),
        p_nn = bijector(Uniform(-Inf, +Inf))) # For neural network parameters
u0_bij = bijector(Uniform(1e-6, 1.0))

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LossLikelihood()
loss_param_prior(p) = norm(p.p_nn) * 1e-3 # regularization on neural network parameters
infprob = InferenceProblem(model_hybrid, model_hybrid.mp.p; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            # loss_param_prior = loss_param_prior,
                            p_bij, u0_bij)

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
                    batchsizes = [5, 5],
                    adtype=Optimization.AutoZygote(), 
                    epochs=[800, 200], 
                    tsteps = tsteps_days_training,
                    optimizers = [OptimizationOptimisers.Adam(eta = 1e-3), 
                                OptimizationOptimisers.Adam(1e-4)],
                    verbose_loss = true,
                    info_per_its = 100,
                    multi_threading = false) # Minimum loss for all batches: 3027.13037109375

# Simulation with inferred parameters
sim_res_inf = simulate(model_hybrid, p = res_inf.p_trained, u0=data[:, 2])
ax_full_pred = Plots.plot(sim_res_inf, title="Full length prediction after training", ylim=(0, 1))

mr_hybrid = hcat(model_hybrid.mortality.(tsteps_days, Ref(res_inf.p_trained))...)'
ax_mortality = plot(mr_hybrid, title="Mortality rates (hybrid model)", labels=["m_A" "m_M"], xlabel="Days", ylabel="Mortality rate")

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
    ylim = (0, 1)
)

myplot = Plots.plot(ax_data, ax_data_pred, ax_full_pred, ax_mortality; layout = @layout([a; b; c; d]), size = ( 800, 1200))
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