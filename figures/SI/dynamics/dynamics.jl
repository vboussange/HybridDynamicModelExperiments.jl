cd(@__DIR__)

using HybridDynamicModels
using HybridDynamicModelExperiments
import HybridDynamicModelExperiments: VaryingGrowthRateModel, Model3SP, Model5SP, Model7SP, InferICs, generate_noisy_data
using OrdinaryDiffEqTsit5
using Random
using Lux

include("../../format.jl")

plt.rcParams["font.size"] = 14

function generate_data(
        ::HybridGrowthRateModel; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [1.0])

    u0_true = [0.5, 0.8, 0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), VaryingGrowthRateModel(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, (; H = p_true.H, q = p_true.q, r = p_true.r[2:end], A = p_true.A)  # only estimating A and r in hybrid model
end

function generate_data(model::Model3SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (H = [1.24, 2.5],
                q = [4.98, 0.8],
                r = [1.0, -0.4, -0.08],
                A = [1.0])

    u0_true = [0.77, 0.060, 0.945]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function generate_data(model::Model5SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (ω = [0.2],
        H = [2.89855, 7.35294, 2.89855, 7.35294],
        q = [1.38, 0.272, 1.38, 0.272],
        r = [1.0, -0.15, -0.08, 1.0, -0.15],
        A = [1.0, 1.0])

    u0_true = [0.77, 0.060, 0.945, 0.467, 0.18]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

function generate_data(model::Model7SP; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (ω = [0.2],
        H = [2.89855, 7.35294, 8.0, 2.89855, 7.35294, 12.0],
        q = [1.38, 0.272, 1e-1, 1.38, 0.272, 5e-2],
        r = [1.0, -0.15, -0.08, 1.0, -0.15, -0.01, -0.005],
        A = [1.0, 1.0])

    u0_true = [0.77, 0.060, 0.945, 0.467, 0.18, 0.14, 0.18]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), model; alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end

tsteps = range(500e0, step = 4, length = 111)
tspan = (tsteps[1], tsteps[end])
tsteps_plot = range(tsteps[1], tsteps[end], length = 1000)
noise = 0.4

fixed_params = (alg = Tsit5(),
    tspan = tspan,
    abstol = 1e-4,
    reltol = 1e-4,
    verbose = false,
    maxiters = 50_000,
    batchsize = 10,
    n_epochs = 3000,
    rng = Random.MersenneTwister(1234)
)


fig, axs = plt.subplots(4, 1, figsize = (9, 9), sharex = true)

models = [Model3SP(), HybridGrowthRateModel(), Model5SP(), Model7SP()]
model_names = [L"\mathcal{M}_3", "Time-varying forcing three-species model variant", L"\mathcal{M}_5", L"\mathcal{M}_7"]
for (i, (model, name)) in enumerate(zip(models, model_names))
    ax = axs[i-1]
    ax.set_title(name, fontsize=14)
    
    # Generate training data at tsteps
    training_data, _ = generate_data(model; fixed_params..., tsteps)
    training_data = generate_noisy_data(training_data, noise, fixed_params.rng)
    true_data, _ = generate_data(model; fixed_params..., tsteps = tsteps_plot)
    
    n_species = size(true_data, 1)
    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    
    for j in 1:n_species
        # Plot true trajectory
        ax.plot(tsteps_plot, true_data[j, :], color = colors[j], linewidth = 2, label = j == 1 ? "True trajectory" : nothing)
        
        # Plot training data points
        ax.scatter(tsteps, training_data[j, :], color = colors[j], s = 20, marker = "o", label = j == 1 ? "Training data" : nothing)
    end

    # ax.set_yscale("log")
    i == 4 && ax.set_xlabel("Time", fontsize=14)
end

# Create a single legend for all subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=14)
fig.supylabel("Species abundance", fontsize = 14)
fig.tight_layout()
display(fig)
fig.savefig("dynamics.pdf", dpi = 300)
