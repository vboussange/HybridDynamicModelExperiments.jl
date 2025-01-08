#=
Stacking up parameter error, forecast error and inference time for all models
Version following refactored scripts (revision 1.2)
=#
cd(@__DIR__)
using Graphs
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using SparseArrays
using ComponentArrays
using SciMLSensitivity

using JLD2
using Distributions
using Bijectors
using DataFrames
using Dates

using PiecewiseInference
import PiecewiseInference: AbstractODEModel

include("../../format.jl")
include("../../../src/loss_fn.jl")
include("../../../src/3sp_model.jl")
include("../../../src/5sp_model.jl")
include("../../../src/7sp_model.jl")
include("../../../src/utils.jl")
include("../../../src/plotting.jl")

result_name_3sp = "../../../scripts/inference_3sp_model/distributed/results/2023-12-02/3sp_model_sims_partial_obs.jld2"


# redefining validate so that we do not get inf values
# if forecasted data is negative, we set it to an arbitrarily low value
import PiecewiseInference.validate
function validate(infres::InferenceResult, ode_data, true_model::AbstractODEModel; length_horizon = nothing)
        loss_likelihood = infres.infprob.loss_likelihood
        tsteps = true_model.mp.kwargs[:saveat]
        mystep = tsteps[2]-tsteps[1]
        ranges = infres.ranges
        isnothing(length_horizon) && (length_horizon = length(ranges[1]))
        tsteps_forecast = range(start = tsteps[end]+mystep, step = mystep, length=length_horizon)
    
        forcasted_data = forecast(infres, tsteps_forecast) |> Array
        true_forecasted_data = simulate(true_model; 
                                        u0 = ode_data[:,ranges[end][1]],
                                        tspan = (tsteps[ranges[end][1]], tsteps_forecast[end]), 
                                        saveat = tsteps_forecast) |> Array
    
        forcasted_data[forcasted_data .<= 0.] .= 1e-3
        loss_likelihood(forcasted_data, true_forecasted_data, nothing)
end

df_result = tap_results(result_name_3sp)
noises = [0.1, 0.2, 0.3]
spread = 0.7 #spread of box plots
legend = true

# println(df_results)
fig, axs = subplots(1, 3, figsize = (6,3))

   
gdf_results = groupby(df_result, [:group_size, :noise])
ax = axs[0]
color_palette = ["tab:blue", "tab:red", "tab:green"]

ylab = "Parameter error"
boxplot_byclass(gdf_results, ax; 
        xname = :group_size,
        yname = :par_err_median, 
        xlab = "", 
        ylab, 
        yscale = "linear", 
        classes = noises, 
        classname = :noise, 
        spread, 
        color_palette,
        legend=false)
fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)



ax = axs[1]
ylab = "Forecast error"
boxplot_byclass(gdf_results, ax; 
        xname = :group_size,
        yname = :val, 
        xlab = "", 
        ylab, 
        yscale = "linear", 
        classes = noises, 
        classname = :noise, 
        spread, 
        color_palette,
        legend=false)
ax.set_facecolor("none")
display(fig)
ax.set_xlabel("Segment size")
ax.set_ylim(-2, 150)


ax = axs[2]
gdf = groupby(df_result, :group_size)
y = [df.time for df in gdf]
x = sort!(unique(df_result.group_size)) .|> Int64
boxplot(ax; y, positions = 1:length(x), color = "tab:gray")
ylab = "Simulation time (s)"
ax.set_ylabel(ylab)
# ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
ax.set_xticks(1:length(x))
ax.set_xticklabels(x, rotation=45)
fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)

labels = ["noise r = $n" for n in noises]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:length(noises)],
        bbox_to_anchor=(0.55, 1.1),
        ncol=3,
        fancybox=true,)
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)

fig.savefig("partial_obs.pdf", dpi = 300, bbox_inches="tight")