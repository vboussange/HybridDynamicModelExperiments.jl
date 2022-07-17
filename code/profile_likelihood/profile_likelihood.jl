#=
Plotting profile likelihood 
for McKann model
=#

cd(@__DIR__)

using FileIO, JLD2
using Statistics
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using LaTeXStrings
using UnPack, ProgressMeter
using MiniBatchInference
using Revise
using PyPlot

include("../model/composable_ecosystem_model.jl")
# include("../parameter_inference/3-species-model/inference_exploration/parse_utils.jl")

#################
### plotting ####
#################

u0_true = [0.5,0.8,0.5]
name_scenario = "3-species-model_McKann_simple_minibatch_allsp"
noise = 0.5
step = 4
datasize = 100
group_size = 6

verbose = true
info_per_its = 200
plot_loss = false
loop = true

alg = Tsit5()
threshold = -1e99 # threshold for stopping optimisation
sensealg = ForwardDiffSensitivity()
abstol = 1e-6
reltol = 1e-6

p_labs = [L"x_c", L"x_p", L"y_c", L"y_p", L"R_0", L"C_0"]

# initialising parameters
em_true = EcosystemModelMcKann(x_c = 0.4, 
                                x_p = 0.09, 
                                y_c = 2.01, 
                                y_p = 5.00, 
                                R_0 = 0.16129, 
                                C_0 = 0.5)

p_true = em_true.p[2]

# plotting the likelihood
fig, axs = subplots(2,2, figsize=(7,7))

for (i,datasize) in enumerate([50,100,200,400])
    ax = axs[i]
    tsteps = range(500., step=step, length = datasize) # we discard transient dynamics, only considering dynamics from 500 to 700
    tspan = (0.,tsteps[end])

    prob = ODEProblem(em_true, u0_true, tspan, em_true.p) # needed for generating data 
    sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
    # using Plots
    # Plots.plot(sol_data)
    ## Ideal data
    data_set = Array(sol_data)

    prob_long = ODEProblem(em_true, u0_true, (0., 1000.), em_true.p) # needed for generating a long time series to obtain variance of variables
    long_data = solve(prob_long, alg, saveat = 500.:1000., abstol=abstol, reltol=reltol, sensealg = sensealg)

    # Data with noise
    std_noise = reshape(std(long_data, dims=2), :) .* noise
    data_set_w_noise = data_set + randn(size(data_set)...)  .* std_noise

    loss_fn(data, pred) = sum((data-pred).^2)
    # normal loss
    function naive_loss(p)
        prob_loss = remake(prob, tspan = (tsteps[1],tsteps[end]), u0 = p[1:3], p = p[4:end], )
        sol = solve(prob_loss, 
                    alg, 
                    saveat = tsteps, 
                    abstol=abstol, 
                    reltol=reltol, 
                    sensealg = sensealg,)
        return loss_fn(data_set_w_noise, sol), sol
    end

    # normal loss
    function EIML_loss(p)
        prob_loss = remake(prob, tspan = (tsteps[1],tsteps[end]), u0 = p[1:3], p = p[4:end],)
        loss, group_predictions = minibatch_loss(p, data_set_w_noise, tsteps, prob_loss,loss_fn, alg, ranges, continuity_term=0.)
        return loss, group_predictions
    end

    update_x_p!(θ, x_p) = 

    # checking the loss function
    if false
        θ = [data_set_w_noise[:,1]; em_true.p]
        l, sol = naive_loss(θ)

        figure()
        plot(Array(sol)')
        plot(data_set', linestyle = "--")
        gcf()
    end

    # naive loss
    θ = [data_set[:,1]; em_true.p]

    # EIML loss
    ranges = MiniBatchInference._get_ranges(group_size, datasize)
    u0s_init_EIML = reshape(data_set[:,first.(ranges),:],:)
    θ_EIML = [u0s_init_EIML;em_true.p]

    x_ps = 0.05:0.001:0.19
    p_likelihood = []
    p_likelihood_EIML = []

    for x_p in x_ps
        θ[5] = x_p
        θ_EIML[length(ranges) * 3 + 2] = x_p
        push!(p_likelihood, naive_loss(θ)[1])
        push!(p_likelihood_EIML, EIML_loss(θ_EIML)[1])
    end

    ax.set_title("Time horizon T = $(tsteps[end] - tsteps[1])")
    ax.plot(x_ps, p_likelihood, label = i == 1 ? "naive loss function" : nothing)
    ax.plot(x_ps, p_likelihood_EIML, label = i == 1 ? "chunked loss function" : nothing)
    ax.vlines(p_true, 0, maximum(p_likelihood), label = i == 1 ?  "true param" : nothing, linestyle = "--", color = "red")
    ax.set_xlabel(L"p_2")
    ax.set_ylabel(L"L(p_2)")

    @save "plikelihood_datasize=$(datasize).jld2" x_ps p_likelihood_EIML p_likelihood p_true
end
fig.legend(bbox_to_anchor=(0.5, 0.66))
# fig.tight_layout()
display(fig)
# fig.savefig("profile_likelihood.png", dpi=500, bbox_tight="true")