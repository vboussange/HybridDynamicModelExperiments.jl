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

include("../../format.jl")

result_name = "../../../scripts/luxbackend/results/luxbackend_gridsearch_segmentshift_3sp_model_31bde13.jld2"

df_err = load(result_name, "results") # size : (2160, 21)
dropmissing!(df_err, :med_par_err) # size : (1441, 21)

df_err_filtered = df_err[(df_err.noise .== 0.2) .&& (df_err.lr .== 1e-2) .&& (df_err.perturb .== 1.0), :]

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(1, 2, figsize = (6,4), sharex = "col")
    # averaging by nruns
gdf_results = groupby(df_err_filtered, [:shift, :infer_ics])
fig.suptitle("Segment length \$S = 8\$, batch size \$b = 10\$, learning rate \$\\gamma = 0.01\$,\n model \$\\mathcal{M}_3\$, noise level \$r = 0.2\$, perturbation magnitude \$\\epsilon = 1.0\$")

boxplot_byclass(gdf_results, axs[0]; 
        xname = :shift,
        yname = :med_par_err, 
        xlab = "", 
        ylab = "Parameter error", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette = COLORS_BR[[1, length(COLORS_BR)]],
        legend=true,
        link=true)
boxplot_byclass(gdf_results, axs[1]; 
        xname = :shift,
        yname = :forecast_err, 
        xlab = "", 
        ylab = "Forecast error", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette = COLORS_BR[[1, length(COLORS_BR)]],
        legend=false,
        link=true)

fig.supxlabel("Segment shift, S")
axs[0].set_facecolor("none")
axs[1].set_facecolor("none")
fig.tight_layout()
display(fig)

fig.savefig("figure3_segmentshift.pdf", dpi = 300, bbox_inches="tight")