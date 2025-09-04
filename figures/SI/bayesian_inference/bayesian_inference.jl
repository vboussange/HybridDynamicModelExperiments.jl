#=
Stacking up parameter error, forecast error and inference time for all models
Version following refactored scripts (revision 1.2)
=#
cd(@__DIR__)
using Graphs
using LinearAlgebra
using UnPack
using Statistics

using JLD2
using Distributions
using DataFrames
using Dates
import HybridModellingExperiments: boxplot

include("../../format.jl")

result_name_3sp = "../../../scripts/mcmcbackend/results/2025-08-29/mcmcbackend_3sp_model.jld2"
result_name_scaling = "../../../scripts/scaling/results/scaling_7cb2a4a.jld2"
df_result = load(result_name_3sp, "results")

df_scaling_lux = load(result_name_scaling, "results")
nits = 5
df_scaling_lux[!, :time] ./= nits * 1e9 # per iteration, in seconds (originally in ns)
dropmissing!(df_scaling_lux)
df_scaling_lux = flatten(df_scaling_lux, :time)
df_scaling_lux = df_scaling_lux[df_scaling_lux.optim_backend .== "MCMCBackend", :]

spread = 0.7 #spread of box plots
legend = true

# println(df_results)
fig, axs = plt.subplots(1, 3, figsize = (8,3))

gdf_results = groupby(df_result, [:segmentsize], sort=true)
ax = axs[0]

ylab = "Parameter error"
y = [df[:, "med_par_err"] for df in gdf_results]
positions = 1:length(gdf_results)
x = [df[:, "segmentsize"][1] for df in gdf_results]
boxplot(ax; 
        y,
        positions, 
        color = "tab:blue")
ax.set_facecolor("none")
ax.set_xticklabels(x)
ax.set_ylabel(ylab)
display(fig)

ax = axs[1]
ylab = "Forecast error"
y = [df[:, "forecast_err"] for df in gdf_results]
boxplot(ax; 
        y,
        positions, 
        color = "tab:blue")
ax.set_facecolor("none")
ax.set_ylabel(ylab)
ax.set_yscale("log")
ax.set_xticklabels(x)
ax.set_xlabel("Segment size")
display(fig)

ax = axs[2]
ylab = "Simulation time\nper epoch (s)"
gdf_results = groupby(df_scaling_lux, [:segmentsize, :infer_ics])

boxplot_byclass(gdf_results, ax; 
        xname = :segmentsize,
        yname = :time, 
        xlab = "", 
        ylab, 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette,
        legend=false)

ax.set_facecolor("none")
ax.set_ylabel(ylab)
ax.set_yscale("log")
display(fig)

labels = ["ICs inferred", "ICs not inferred"]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:2],
        bbox_to_anchor=(0.55, 1.1),
        ncol=3,
        fancybox=true,)

fig.set_facecolor("none")
fig.tight_layout()


_let = ["A","B","C","D"]
for (i,ax) in enumerate(axs)
    _x = -0.2
    ax.text(_x, 0.99, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
        zorder = 199
    )
end

display(fig)
# TODO: you could add two more subplots with posterior for each params, as well as data retrodiction

fig.savefig("bayesian_inference.pdf", dpi = 300, bbox_inches="tight")