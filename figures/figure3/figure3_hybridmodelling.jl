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

include("../format.jl")

tsteps = range(500e0, step = 4, length = 100)
segmentsizes = floor.(Int, exp.(range(log(2), log(100), length = 6)))
nsegments = [length(tokens(tokenize(SegmentedTimeSeries(tsteps, segmentsize = s)))) for s in segmentsizes]

result_name = "../../scripts/luxbackend/results/luxbackend_gridsearch_3sp_5sp_7sp_model_31bde13.jld2"

df_err = load(result_name, "results") # size : (2160, 21)
dropmissing!(df_err, :med_par_err) # size : (1441, 21)

df_err_filtered = df_err[(df_err.noise .== 0.2) .&& (df_err.lr .== 1e-3) .&& (df_err.perturb .== 1e0), :]

gdf_err = groupby(df_err_filtered, :modelname)

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(2, 3, figsize = (6,4), sharex = "col", sharey = "row")
for (i, modelname) in enumerate(["Model3SP", "Model5SP", "Model7SP"])
    # averaging by nruns
    i = i-1
    _df_err = gdf_err[(;modelname)]
    gdf_results = groupby(_df_err, [:segmentsize, :infer_ics])
    i == -1 ? legend = true : legend = false
    ax = axs[0, i]

    i == 0 ? ylab = "Parameter error" : ylab = ""
    boxplot_byclass(gdf_results, ax; 
            xname = :segmentsize,
            yname = :med_par_err, 
            xlab = "", 
            ylab, 
            yscale = "linear", 
            classes = [true, false], 
            classname = :infer_ics, 
            spread, 
            color_palette,
            legend, 
            link=true)
    
    ax.set_facecolor("none")
    fig.tight_layout()
    display(fig)



    ax = axs[1, i]
    i == 0 ? ylab = "Forecast error" : ylab = ""
    boxplot_byclass(gdf_results, ax; 
            xname = :segmentsize,
            yname = :forecast_err, 
            xlab = "", 
            ylab, 
            yscale = "linear", 
            classes = [true, false], 
            classname = :infer_ics, 
            spread, 
            color_palette,
            legend=false,
            link=true)
#     ax.set_ylim(-0.2,2.)
    ax.set_facecolor("none")
    # ax.set_ylim(-0.1,2e0)
    ax.set_yscale("log")

    fig.set_facecolor("none")
    fig.tight_layout()
    display(fig)
end

labels = ["ICs inferred", "ICs not inferred"]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:2],
        bbox_to_anchor=(0.55, 1.05),
        ncol=3,
        fancybox=true,)
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)

fig.savefig("figure3_temp.pdf", dpi = 300, bbox_inches="tight")