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

include("../../format.jl")

result_name_3sp = "../../../scripts/luxbackend/results/2025-08-30/luxbackend_3sp_model.jld2"

df_3sp = load(result_name_3sp, "results")
df_3sp[!, :med_par_err] = abs.(df_3sp[:, :med_par_err])

noise = 0.1
infer_ics = true
df_3sp_filtered = filter(row -> row.noise == noise && row.infer_ics == infer_ics, df_3sp)

classname = :lr
classes = sort!(unique(df_3sp_filtered[:, classname]))

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(2, 1, figsize = (6,6), sharex = "col", sharey = "row")
# averaging by nruns
gdf_results = groupby(df_3sp_filtered, [:segmentsize, classname])
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
        legend=true)
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
        classname = :lr, 
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