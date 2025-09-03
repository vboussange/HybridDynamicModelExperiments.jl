#=
Generating figure 3 of main manuscript. 

To get full figure, you need to compile `fig_3_tex/figure_3.tex`, which overlays
graphical illustration of foodwebs on top of the figure produced by this script.
=#
cd(@__DIR__)
using UnPack
using Statistics
using JLD2
using Distributions
using DataFrames
using Dates
using HybridModelling
using HybridModellingExperiments: boxplot_byclass

include("../../format.jl");

result_name = "../../../scripts/luxbackend/results/luxbackend_hybridfuncresp_model_07d2781.jld2"

df = load(result_name, "results")
dropmissing!(df, :med_par_err)

weight_decay = 1e-5
df_filtered = filter(row -> row.weight_decay == weight_decay, df)

classname = :infer_ics
classes = [true, false]

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(2, 1, figsize = (6,6), sharex = "col", sharey = "row")
# averaging by nruns
gdf_results = groupby(df_filtered, [:segmentsize, classname])
ax = axs[0]

boxplot_byclass(gdf_results, ax;
        xname = :segmentsize,
        yname = :med_par_err, 
        xlab = "Segment size", 
        ylab = "Parameter error", 
        yscale = "linear", 
        classes, 
        classname, 
        spread, 
        color_palette,
        legend=false)
fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)

ax = axs[1]
boxplot_byclass(gdf_results, ax; 
        xname = :segmentsize,
        yname = :forecast_err, 
        xlab =  "Segment size", 
        ylab = "Forecast error", 
        yscale = "linear", 
        classes = classes, 
        classname, 
        spread, 
        color_palette, 
        legend=false)
ax.set_yscale("log")
ax.set_facecolor("none")
display(fig)

fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)

# Ms = [Model3SP, Model5SP, Model7SP]
# model_names = [L"\mathcal{M}_3", L"\mathcal{M}_5", L"\mathcal{M}_7"]
# @assert all([df.res[1].infprob.m isa M for (df,M) in zip(df_result_arr, Ms)])

fig.savefig("lr.pdf", dpi = 300, bbox_inches="tight")