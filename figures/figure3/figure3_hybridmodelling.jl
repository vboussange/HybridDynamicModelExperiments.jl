#=
Generating figure 3 of main manuscript. 

To get full figure, you need to compile `fig_3_tex/figure_3.tex`, which overlays
graphical illustration of foodwebs on top of the figure produced by this script.
=#
cd(@__DIR__)
using UnPack
using Statistics
using SparseArrays
using JLD2
using Distributions
using DataFrames
using Dates
using HybridModelling
using HybridModellingExperiments

include("../format.jl")

result_name_3sp = "../../scripts/luxbackend/results/2025-08-30/luxbackend_3sp_model.jld2"
# result_name_5sp = "../../scripts/inference_5sp_model/distributed/results/2023-11-23/5sp_model_sims_only_backdiff.jld2"
# result_name_7sp = "../../scripts/inference_7sp_model/distributed/results/2023-11-23/7sp_model_sims_only_backdiff.jld2"


# Stacking up parameter error, forecast error and inference time for all models
df_result_arr = [load(rn, "results") for rn in [result_name_3sp]]
noises = 0.2
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

Ms = [Model3SP, Model5SP, Model7SP]
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