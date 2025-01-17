#= 
Inference simulations wuth `Model3SP` with different optimisers.
=#

cd(@__DIR__)

using FileIO, JLD2
using Graphs, LinearAlgebra, UnPack
using OrdinaryDiffEq, Statistics, SparseArrays, ComponentArrays
using PythonPlot, SciMLSensitivity, PiecewiseInference
using OptimizationOptimisers, OptimizationOptimJL, Dates, Bijectors, PythonCall, Random
using DataFrames
Random.seed!(2)
plt = pyimport("matplotlib.pyplot")

include("../../src/3sp_model.jl")
include("../../src/loss_fn.jl")
include("../../src/utils.jl")
include("../../src/plotting.jl")

function initialize_inference_problem(model, p_true, loss_likelihood, perturb=1f0)
    u0_bij = bijector(Uniform(1e-3, 5.0))

    # Parameter constraints
    distrib_param = NamedTuple([
        dp => Product([Uniform(sort([(1.0-perturb/2.0) * k, (1.0+perturb/2.0) * k])...) for k in p_true[dp]])
        for dp in keys(p_true)
    ])

    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)]) |> ComponentArray

    infprob = InferenceProblem(model, p_init; 
                               loss_u0_prior=loss_likelihood,
                               loss_likelihood=loss_likelihood,
                               p_bij, u0_bij)
    return infprob
end

# Simulation parameters
noise = 0.1
data_params = (datasize=100, step=4)
tsteps = range(500.0, step=data_params.step, length=data_params.datasize)
tspan = (0.0, last(tsteps))

model_params = (
    alg=Tsit5(),
    abstol=1e-4,
    reltol=1e-4,
    tspan=tspan,
    sensealg=ForwardDiffSensitivity(),
    saveat=tsteps,
    maxiters=50_000,
)

# True model parameters
u0_true = Float32[0.77, 0.060, 0.945]
p_true = ComponentArray(
    H=Float32[1.24, 2.5],
    q=Float32[4.98, 0.8],
    r=Float32[1.0, -0.4, -0.08],
    A=Float32[1.0]
)

# Initialize model and generate data
mp = ModelParams(;tspan, u0=u0_true, saveat=tsteps, model_params...)
model = Model3SP(mp)
data = simulate(model, u0=u0_true, p=p_true) |> Array
data_w_noise = data .* exp.(noise * randn(size(data)))

plotting && display(plot_time_series(data_w_noise, model))

inference_params = (
    adtype=Optimization.AutoForwardDiff(),
    data=data,
    tsteps=tsteps,
    epochs=[2000],
    verbose_loss=true,
    info_per_its=100,
    multi_threading=false
)

# Perform inference for different perturbation setups
infprob = initialize_inference_problem(model, p_true, LossLikelihood())

results_df = DataFrame(
    group_size = Int[],
    time       = Float64[],
    res        = Any[],
    optim_name = String[],
    ps = Any[],
    perr_all = Any[],
    perr = Float64[]
)

for (opt, nameopt) in zip([OptimizationOptimJL.LBFGS(), OptimizationOptimisers.Adam(1e-2), OptimizationOptimisers.AdaGrad(1e-2)], ["LBFGS", "Adam", "AdaGrad"])
    for group_size in [11, size(data_w_noise, 2)]
        ps = []
        callback = (p, u0s, l) -> push!(ps, p)
        stats = @timed inference(infprob; group_size, cb=callback, optimizers = [opt], inference_params...)
        res = stats.value
        perr_all = [median(abs.((p_true .- p) ./ p_true)) for p in ps]
        push!(results_df, (group_size, stats.time, res, nameopt, ps, perr_all, perr_all[end]))
    end
end
save_results(@__FILE__; results=results_df)


if true
    using DataFrames: groupby
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=true)
    group_sizes = unique(results_df.group_size)
    for (i, gsize) in enumerate(group_sizes)
        subset = filter(row -> row.group_size == gsize, eachrow(results_df))
        for row in subset
            axs[i-1].plot(1:length(row.perr_all), row.perr_all, label= i == 1 ? row.optim_name : nothing)
        end
        axs[i-1].set_title("Param Error vs Iteration (group_size = $gsize)")
        axs[i-1].set_xlabel("Iteration")
        axs[i-1].set_ylabel("Absolute Relative Error")
        axs[i-1].legend(loc="upper right")
    end
    display(fig)
end