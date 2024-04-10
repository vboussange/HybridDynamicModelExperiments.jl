#=
Generating figure 3 of main manuscript. 

To get full figure, you need to compile `fig_3_tex/figure_3.tex`, which overlays
graphical illustration of foodwebs on top of the figure produced by this script.
=#
cd(@__DIR__)
using Graphs
using ParametricModels
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics
using SparseArrays
using ComponentArrays
using SciMLSensitivity
using PiecewiseInference
using JLD2
using Distributions
using Bijectors
using DataFrames
using Dates

include("../format.jl")
include("../../src/loss_fn.jl")
include("../../src/3sp_model.jl")
include("../../src/5sp_model.jl")
include("../../src/7sp_model.jl")
include("../../src/utils.jl")
include("../../src/plotting.jl")

result_name_3sp = "../../scripts/inference_3sp_model/distributed/results/2023-11-23/3sp_model_sims_only_backdiff.jld2"
result_name_5sp = "../../scripts/inference_5sp_model/distributed/results/2023-11-23/5sp_model_sims_only_backdiff.jld2"
result_name_7sp = "../../scripts/inference_7sp_model/distributed/results/2023-11-23/7sp_model_sims_only_backdiff.jld2"

color_palette = ["tab:blue", "tab:red", "tab:green"]

# redefining validate so that we do not get inf values
# if forecasted data is negative, we set it to an arbitrarily low value
import PiecewiseInference.validate
function validate(infres::InferenceResult, ode_data, true_model::AbstractModel; length_horizon = nothing)
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

# Stacking up parameter error, forecast error and inference time for all models
df_result_arr = [tap_results(rn) for rn in [result_name_3sp, result_name_5sp, result_name_7sp]]
noises = [0.1, 0.2, 0.3]
spread = 0.7 #spread of box plots

fig, axs = subplots(3, 3, figsize = (6,6), sharex = "col", sharey = "row")
for (i,df_results) in enumerate(df_result_arr)
    # averaging by nruns
    i = i-1
    gdf_results = groupby(df_results, [:group_size, :noise])
    i == -1 ? legend = true : legend = false
    ax = axs[0, i]

    i == 0 ? ylab = "Parameter error" : ylab = ""
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
            legend)
    fig.set_facecolor("none")
    ax.set_facecolor("none")
    fig.tight_layout()
    display(fig)



    ax = axs[1, i]
    i == 0 ? ylab = "Forecast error" : ylab = ""
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
    ax.set_ylim(-10,100)
    ax.set_facecolor("none")
    display(fig)


    ax = axs[2, i]
    gdf = groupby(df_results, :group_size)
    y = [df.time for df in gdf]
    x = sort!(unique(df_results.group_size)) .|> Int64
    boxplot(ax; y, positions = 1:length(x), color = "tab:gray")
    i == 0 ? ylab = "Simulation time (s)" : ylab = ""
    ax.set_ylabel(ylab)
    # ax.set_yscale("log")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Segment size")
    ax.set_xticks(1:length(x))
    ax.set_xticklabels(x, rotation=45)
    fig.set_facecolor("none")
    ax.set_facecolor("none")
    fig.tight_layout()
    display(fig)

end

Ms = [SimpleEcosystemModel3SP, SimpleEcosystemModel5SP, SimpleEcosystemModel7SP]
model_names = [L"\mathcal{M}_3", L"\mathcal{M}_5", L"\mathcal{M}_7"]
@assert all([df.res[1].infprob.m isa M for (df,M) in zip(df_result_arr, Ms)])

labels = ["noise r = $n" for n in noises]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:length(noises)],
        bbox_to_anchor=(0.55, 1.2),
        ncol=3,
        fancybox=true,)
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)

fig.savefig("figure3_temp.pdf", dpi = 300, bbox_inches="tight")