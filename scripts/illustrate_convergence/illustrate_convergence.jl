#= 
Inference simulations with `Model3SP` for illustrating convergence with different learning strategy in figure 1.
Three learning strategies:
1. No segmentation
2. No segmentation with initial parameter values close to true parameter values
3. Segmentation
=#

cd(@__DIR__)

using FileIO, JLD2
using Graphs, LinearAlgebra, UnPack
using OrdinaryDiffEq, Statistics, SparseArrays, ComponentArrays
using PythonPlot, SciMLSensitivity, PiecewiseInference
using OptimizationFlux, OptimizationOptimJL, Dates, Bijectors, PythonCall, Random
Random.seed!(2)
plt = pyimport("matplotlib.pyplot")

include("../../src/3sp_model.jl")
include("../../src/loss_fn.jl")
include("../../src/utils.jl")
include("../../src/plotting.jl")

function initialize_inference_problem(model, p_true, loss_likelihood, p_init, perturb=1f0)
    u0_bij = bijector(Uniform(1e-3, 5.0))

    # Parameter constraints
    distrib_param = NamedTuple([
        dp => Product([Uniform(sort([(1.0-perturb/2.0) * k, (1.0+perturb/2.0) * k])...) for k in p_true[dp]])
        for dp in keys(p_true)
    ])

    p_bij = NamedTuple([dp => bijector(distrib_param[dp]) for dp in keys(distrib_param)])

    infprob = InferenceProblem(model, p_init; 
                               loss_u0_prior=loss_likelihood,
                               loss_likelihood=loss_likelihood,
                               p_bij, u0_bij)
    return infprob
end

function get_initial_parameters(p_true, perturb)
    distrib_param = NamedTuple([
        dp => Product([Uniform(sort([(1.0-perturb/2.0) * k, (1.0+perturb/2.0) * k])...) for k in p_true[dp]])
        for dp in keys(p_true)
    ])
    p_init = NamedTuple([k => rand(distrib_param[k]) for k in keys(distrib_param)])
    p_init = ComponentArray(p_init) .|> eltype(data)
end

# Simulation parameters
plotting = true
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
    optimizers=[Adam(1e-2)],
    epochs=[5000],
    verbose_loss=true,
    info_per_its=100,
    multi_threading=false
)

# Inference execution
results = Dict{String, Any}()
results["p_true"] = p_true

perturb = 1.0
p_init = get_initial_parameters(p_true, perturb)
p_init_small_pert = get_initial_parameters(p_true, perturb / 10)

# Perform inference for different perturbation setups
infprob = initialize_inference_problem(model, p_true, LossLikelihood(), p_init, perturb)
infprob_small_pert = initialize_inference_problem(model, p_true, LossLikelihood(), p_init_small_pert, perturb)

@info "Performing batch inference"
θs = []
callback = (p_trained, losses, pred, ranges) -> push!(θs, p_trained)
@time res = inference(infprob; group_size=11, cb=callback, inference_params...)
results["batch_inference"] = θs

@info "Performing naive inference"
θs = []
@time res = inference(infprob; group_size=100, cb=callback, inference_params...)
results["naive_inference"] = θs

@info "Performing naive inference with small perturbation"
θs = []
@time res = inference(infprob_small_pert; group_size=100, cb=callback, inference_params...)
results["naive_inference_small_p"] = θs

if plotting
    p_err_dict = Dict()
    fig, ax = plt.subplots()
    for k in ["naive_inference",  "naive_inference_small_p", "batch_inference", ]
        p_true = results["p_true"]
        perr = [median(abs.((p_true .- p) ./ p_true)) for p in results[k]]
        p_err_dict[k] = perr
    end
    ax.plot(1:length(p_err_dict["batch_inference"])-2, p_err_dict["batch_inference"][1:end-2], label = L"L_{\mathcal{M}}^\star", linestyle = linestyles[2],c="tab:orange")
    ax.plot(1:length(p_err_dict["naive_inference"])-2, p_err_dict["naive_inference"][1:end-2], label = L"L_{\mathcal{M}}", linestyle = linestyles[1],c="tab:blue")
    ax.plot(1:length(p_err_dict["naive_inference_small_p"])-2, p_err_dict["naive_inference_small_p"][1:end-2], label = L"L_{\mathcal{M}}, \theta_0 = \Tilde{\theta} + \delta\theta", linestyle = linestyles[3],c="tab:green")
    ax.legend(loc="upper right")
    # ax.set_yscale("log")
    ax.set_ylabel(L"|\nicefrac{(\hat p -\Tilde{p})}{\Tilde{p}}|")

    ax.set_xlabel("Epochs")
    display(fig)
end

save(split(@__FILE__, ".")[1] * ".jld2", results)
println("Results saved successfully.")
