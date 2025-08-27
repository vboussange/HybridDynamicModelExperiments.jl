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
import HybridModellingExperiments: boxplot

include("../../format.jl")

result_name_3sp = "../../../scripts/benchmark/results/2025-08-26/benchmark_test_mcmc.jld2"
df_result = load(result_name_3sp, "results")

spread = 0.7 #spread of box plots
legend = true

# println(df_results)
fig, axs = plt.subplots(1, 3, figsize = (6,3))


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
ax.set_xlabel("Segment size")
display(fig)

ax = axs[2]
ylab = "Simulation time (s)"
y = [df[:, "segmentsize"] for df in gdf_results]
boxplot(ax; 
        y,
        positions, 
        color = "tab:grey")
ax.set_facecolor("none")
ax.set_ylabel(ylab)
ax.set_yscale("log")
display(fig)

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