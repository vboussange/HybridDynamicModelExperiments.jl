#=

Continuity analysis of McKann model

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
using Random; Random.seed!(2)
include("../../format.jl")
Line2D = matplotlib.lines.Line2D #used for legend

include("../../../code/model/composable_ecosystem_model.jl")
# include("../parameter_inference/3-species-model/inference_exploration/parse_utils.jl")

##################################
### sensitivity to parameters ####
##################################

u0_init = [0.5,0.8,0.5]
name_scenario = "3-species-model_McKann_simple_minibatch_allsp"

verbose = true
info_per_its = 200
plot_loss = false
loop = true

alg = Tsit5()
threshold = -1e99 # threshold for stopping optimisation
sensealg = ForwardDiffSensitivity()
abstol = 1e-6
reltol=1e-6

p_labs = [L"x_c", L"x_p", L"y_c", L"y_p", L"R_0", L"C_0"]

# initialising parameters
em_true = EcosystemModelMcKann(x_c = 0.4, 
                                x_p = 0.071, 
                                y_c = 2.01, 
                                y_p = 5.00, 
                                R_0 = 0.16129, 
                                C_0 = 0.5)
p_pert = em_true.p .+ 0.1 * em_true.p .* randn(length(em_true.p))

datasize = 40
step = 1
tsteps = range(500., 900., step=step) # we discard transient dynamics, only considering dynamics from 500 to 700
tspan = (tsteps[1],tsteps[end])

prob = ODEProblem(em_true, u0_init, tspan, em_true.p) # needed for generating data 
sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
u0_true = sol_data[:,1]

# calculating sensitivities, for two problems where one has a parameter that is slightly perturbed
prob_single = ODELocalSensitivityProblem(em_true, u0_true, tspan, em_true.p)
prob_single_perturbed = ODELocalSensitivityProblem(em_true, u0_true, tspan, p_pert)

sol_single = solve(prob_single, saveat=tsteps, alg = Tsit5())
sol_single_perturbed = solve(prob_single_perturbed, saveat=tsteps, alg = Tsit5())

x_single, dp = extract_local_sensitivities(sol_single,)
x_single_perturbed, dp_perturbed = extract_local_sensitivities(sol_single_perturbed,)

Nt = length(dp[1][1,:]) # nb of time steps
Nstate = length(dp[1][:,1]) # nb of state variables
Nparam = length(dp[:,1]) # nb of parameters


#=
Plotting sensitivities for perturbed and unperturbed problem.
=#
color_palette = ["tab:red", "tab:blue", "tab:green"]
ylabels = [L"\frac{\partial R(t)}{\partial x_p}", L"\frac{\partial C(t)}{\partial x_p}", L"\frac{\partial P(t)}{\partial x_p}"]
fig, axs = plt.subplots(3,1, figsize=(5.2,5))
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]

for i in 1:length(axs)
    axs[i].plot(tsteps, (dp[2][i,:]), label = "p", color = color_palette[i])
    axs[i].plot(tsteps, (dp_perturbed[2][i,:]), label = L"p + \delta p", linestyle = "--", color = color_palette[i])
    axs[i].set_ylabel(ylabels[i])
end
axs[1].legend()
for ax in axs
    ax.set_xlabel("Time");
    # ax.set_yscale("symlog")
end

fig.tight_layout()
display(fig)

#=
Plotting trjaectories for perturbed and unperturbed problem.
=#
ylabels = [L"R(t)", L"C(t)", L"P(t)"]
fig, axs = plt.subplots(3,1, figsize=(5.2,4))
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]
for i in 1:length(axs)
    axs[i].plot(tsteps .- 500., (x_single[i,:]), color = color_palette[i])
    axs[i].plot(tsteps .- 500., (x_single_perturbed[i,:]), linestyle = "--", color = color_palette[i])
    axs[i].set_ylabel(ylabels[i])
end
fig.legend(ncol=2,
            handles=[
                    Line2D([0], [0], color="grey", label=L"x(t)"),
                    Line2D([0], [0], color="grey", linestyle="--", label=L"x_{\delta x_0}(t)"),
            ],
            loc="upper center",
            bbox_to_anchor = (0.5,1.1), 
            fontsize=10)
fig.tight_layout()
display(fig)
fig.savefig("perturbed_p.png", dpi = 300, bbox_inches="tight")

##################################
### sensitivity to ICs        ####
##################################

datasize = 40
step = 1
tsteps = range(500., 900., step=step) # we discard transient dynamics, only considering dynamics from 500 to 700
tspan = (tsteps[1],tsteps[end])

prob = ODEProblem(em_true, u0_init, tspan, em_true.p) # needed for generating data 
sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
u0_true = sol_data[:,1]
u0_perturbed = u0_true + 0.4 * u0_true .* randn(length(u0_true))

# calculating sensitivities, for two problems where one has a parameter that is slightly perturbed
prob_single = ODELocalSensitivityProblem(em_true, u0_true, tspan, em_true.p)
prob_single_perturbed = ODELocalSensitivityProblem(em_true, u0_perturbed, tspan, em_true.p)

sol_single = solve(prob_single, saveat=tsteps, alg = Tsit5())
sol_single_perturbed = solve(prob_single_perturbed, saveat=tsteps, alg = Tsit5())

x_single, dp = extract_local_sensitivities(sol_single,)
x_single_perturbed, dp_perturbed = extract_local_sensitivities(sol_single_perturbed,)

Nt = length(dp[1][1,:]) # nb of time steps
Nstate = length(dp[1][:,1]) # nb of state variables
Nparam = length(dp[:,1]) # nb of parameters

#=
Plotting trjaectories for perturbed and unperturbed problem.
=#
ylabels = [L"R(t)", L"C(t)", L"P(t)"]
fig, axs = plt.subplots(3,1, figsize=(5.2,4))
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]

for i in 1:length(axs)
    axs[i].plot(tsteps .- 500., (x_single[i,:]), color = color_palette[i])
    axs[i].plot(tsteps .- 500., (x_single_perturbed[i,:]), linestyle = "--", color = color_palette[i])
    axs[i].set_ylabel(ylabels[i])
end
fig.legend(ncol=2,
            handles=[
                    Line2D([0], [0], color="grey", label=L"x(t)"),
                    Line2D([0], [0], color="grey", linestyle="--", label=L"x_{\delta x_0}(t)"),
            ],
            loc="upper center",
            bbox_to_anchor = (0.5,1.1),
            fontsize=10)
fig.tight_layout()
display(fig)
fig.savefig("perturbed_ICs.png", dpi = 300, bbox_inches="tight")


#=
Plotting sensitivities for perturbed and unperturbed problem.
=#
color_palette = ["tab:red", "tab:blue", "tab:green"]
ylabels = [L"\frac{\partial R(t)}{\partial x_p}", L"\frac{\partial C(t)}{\partial x_p}", L"\frac{\partial P(t)}{\partial x_p}"]
fig, axs = plt.subplots(3,1, figsize = (5,2,5))

for i in 1:length(axs)
    axs[i].plot(tsteps .- 500., (dp[2][i,:]), label = "p", color = color_palette[i])
    axs[i].plot(tsteps .- 500., (dp_perturbed[2][i,:]), label = L"p + \delta p", linestyle = "--", color = color_palette[i])
    axs[i].set_ylabel(ylabels[i], fontsize = 12)
end
axs[1].legend()
for ax in axs
    ax.set_xlabel("Time");
    # ax.set_yscale("symlog")
end

fig.tight_layout()
# display(fig)