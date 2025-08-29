#=
Short exampling showcasing the fit of a 3 species model.
=#
cd(@__DIR__)
import OrdinaryDiffEqTsit5
using Plots
using Distributions
using Bijectors
using Optimization, OptimizationOptimisers
using SciMLSensitivity
using Dates, DataFrames, CSV
DATA_PATH = joinpath(@__DIR__, "../data/beninca")
include("../src/beninca_model.jl")
include("../src/loss_fn.jl")


# importing data
function load_data()
    data_df = DataFrame(CSV.File(joinpath(DATA_PATH, "beninca_data.csv")))
    data_df[!, 1] = Date.(data_df[:, 1])  # Convert first column to Date
    data_df[!, 2:end] = Float32.(data_df[:, 2:end])
    data_df[!, 2:end] = ifelse.(data_df[:, 2:end] .< 1f-6, 1f-6, data_df[:, 2:end]) # Set values below 1e-6 to 1e-6
    return data_df
end

data_df = load_data()
data = Matrix(data_df[:, 2:end])'
tsteps = data_df[:, 1]
tsteps_days = Dates.date2epochdays.(tsteps) .|> Float32
# NOTE: to convert back to Date, 
# tsteps_reconstructed = Dates.epochdays2date.(round.(Int, tsteps_days))

ax_data = Plots.plot(tsteps, data', title = "Data", labels = reshape(names(data_df)[2:end], 1, :))

# constructing the interpolation for the forcing data
forcing = TrueTempForcing()

axforcing = Plots.plot(tsteps, hcat(forcing.(tsteps_days)...)', title = "Forcing")
Plots.plot(ax_data, axforcing)


# Model metaparameters
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (tsteps_days[1], tsteps_days[end]) # 10 years in days

# B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
# overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
# mussels)
u0_init = Float32[0.103, 0.019, 0.033, 0.040] 

p_init = ComponentArray(
                c_BR = [0.018f0],
                c_AB = [0.049f0],
                c_M = [0.078f0],
                c_AR = [0.021f0],
                m_B = [0.003f0])

# Model initialization with true parameters
model_hybrid = hybrid_beninca_model(ModelParams(;p= p_init,
                            tspan,
                            u0 = u0_init,
                            alg,
                            reltol,
                            abstol,
                            saveat = tsteps_days,
                            verbose = true,
                            maxiters = 50_000,
                            ), forcing)
p_synthetic = ComponentArray(
    c_BR = [0.018f0],
    c_AB = [0.049f0],
    c_M = [0.078f0],
    c_AR = [0.021f0],
    m_B = [0.003f0],
    m_A = [0.011f0],
    m_M = [0.017f0],
    α = [1f-2],
)
forcing_synthetic = SyntheticTempForcing(30.0f0, 20.0f0)
model_synthetic = classic_beninca_model(ModelParams(;p= p_synthetic,
                            tspan,
                            u0 = u0_init,
                            alg,
                            reltol,
                            abstol,
                            saveat = tsteps_days,
                            verbose = true,
                            maxiters = 50_000,
                            ), forcing_synthetic)

mr_synthetic = hcat(model_synthetic.mortality.(tsteps_days, Ref(model_synthetic.mp.p))...)'
plot(mr_synthetic, title="Mortality rates (synthetic model)", labels=["m_A" "m_M"], xlabel="Days", ylabel="Mortality rate")

mr_hybrid = hcat(model_hybrid.mortality.(tsteps_days, Ref(model_hybrid.mp.p))...)'
plot!(mr_hybrid, title="Mortality rates (hybrid model)", labels=["m_A" "m_M"], xlabel="Days", ylabel="Mortality rate")



plot(forcing_synthetic.(tsteps_days), title="Forcing (synthetic model)", labels=["Forcing"], xlabel="Days", ylabel="Temperature")
plot!(hcat(forcing.(tsteps_days)...)', title="Forcing (hybrid model)", labels=["Forcing"], xlabel="Days", ylabel="Temperature")


# Data generation
# TODO: we need to fine-tune the neural net and parameter bijectors to obtain stable simulations
# But first check if the model is well coded by plugging in a simple mortality function
# NOTE: it could be that unstable simulations are the cause of the `inference` failing
sim0 = simulate(model_hybrid, u0 = data[:,1]) |> Array # throws error
ax_sim0 = Plots.plot(tsteps, sim0', title = "Initial guess", labels = ["B₀" "Bᵢ" "A" "M"])
Plots.plot(ax_sim0, ax_data)

p_bij = (c_BR = bijector(Uniform(1f-3, 0.1f0)),
          c_AB = bijector(Uniform(1f-3, 0.1f0)),
          c_M = bijector(Uniform(1f-3, 0.1f0)),
          c_AR = bijector(Uniform(1f-3, 0.1f0)),
          m_B = bijector(Uniform(1f-3, 0.01f0)),
        p_nn = bijector(Uniform(-Inf, +Inf))) # For neural network parameters
u0_bij = bijector(Uniform(1f-6, 1f0))

# Defining inference problem
# Model initialized with perturbed parameters
loss_likelihood = LossLikelihood()

infprob = InferenceProblem(model_synthetic, model_synthetic.mp.p; 
                            loss_u0_prior = loss_likelihood, 
                            loss_likelihood = loss_likelihood, 
                            p_bij, u0_bij)

# debugging
using Zygote
function loss_fn(p)
    return sum(simulate(model, p=p, u0 = data[:,5], tspan=(tsteps_days[1], tsteps_days[10]), saveat = tsteps_days[1:10]))
end
Zygote.gradient(loss_fn, model.mp.p)

# Inference
res_inf = inference(infprob;
                    data=Matrix(data), 
                    group_size = 10, 
                    adtype=Optimization.AutoZygote(), 
                    epochs=[200], 
                    tsteps = tsteps_days,
                    optimizers = [OptimizationOptimisers.Adam(5e-2)],
                    verbose_loss = true,
                    info_per_its = 1,
                    multi_threading = false)

# Simulation with inferred parameters
sim_res_inf = simulate(model, p = res_inf.p_trained, u0=data[:, 1], tspan=(tsteps[1], tsteps[end]))
ax3 = Plots.plot(sim_res_inf, title="Predictions after training", )
Plots.plot(ax, ax2, ax3)