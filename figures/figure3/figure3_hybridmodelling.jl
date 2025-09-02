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
using HybridModellingExperiments: boxplot_byclass, boxplot

include("../format.jl")

result_name_3sp = "../../scripts/luxbackend/results/luxbackend_3sp_model_4f148a8.jld2"
result_name_5sp_7sp = "../../scripts/luxbackend/results/luxbackend_5sp_7sp_model_1c400bd.jld2"
result_name_scaling = "../../scripts/scaling/results/scaling_e19b20d.jld2"

df_3sp = load(result_name_3sp, "results")
df_3sp[!, :med_par_err] = abs.(df_3sp[:, :med_par_err])
df_3sp = df_3sp[df_3sp[:, :lr] .== 0.01, :] # lr should be same for both arrays

df_5sp_7sp = load(result_name_5sp_7sp, "results")

common_cols = names(df_3sp)
df_err = vcat(select(df_3sp, common_cols), select(df_5sp_7sp, common_cols))
gdf_err = groupby(df_err, :modelname)

df_scaling_lux, nits = load(result_name_scaling, "lux_results", "nits")
# df_scaling_mcmc = load(result_name_scaling, "mcmc_results")
df_scaling_lux[!, :time] ./= nits # per iteration
gdf_scaling_lux = groupby(df_scaling_lux, :modelname)

noise = 0.2
df_err_filtered = filter(row -> row.noise == noise, df_err) # TODO: filter by lr too

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(3, 3, figsize = (6,6), sharex = "col", sharey = "row")
for (i, modelname) in enumerate(["Model3SP", "Model5SP", "Model7SP"])
    # averaging by nruns
    i = i-1
    df_err = gdf_err[(;modelname)]
    gdf_results = groupby(df_err, [:segmentsize, :infer_ics])
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
            legend)
    fig.set_facecolor("none")
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
            legend=false)
#     ax.set_ylim(-0.2,2.)
    ax.set_facecolor("none")
    display(fig)
    ax.set_ylim(1e-3,1e2)
    ax.set_yscale("log")

#     scaling
    ax = axs[2, i]
    i == 0 ? ylab = "Simulation time\nper epoch (s)" : ylab = ""
    df_scaling_lux = gdf_scaling_lux[(;modelname)]
    gdf_results = groupby(df_scaling_lux, [:segmentsize, :infer_ics])
    # y = [df.time for df in gdf]
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
    ax.set_ylabel(ylab)
    # ax.set_yscale("log")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Segment size")
    x = sort!(unique(df_scaling_lux.segmentsize)) .|> Int64
    ax.set_xticks(1:length(x))
    ax.set_xticklabels(x, rotation=45)
    fig.set_facecolor("none")
    ax.set_facecolor("none")
    fig.tight_layout()
    display(fig)
end

# Ms = [Model3SP, Model5SP, Model7SP]
# model_names = [L"\mathcal{M}_3", L"\mathcal{M}_5", L"\mathcal{M}_7"]
# @assert all([df.res[1].infprob.m isa M for (df,M) in zip(df_result_arr, Ms)])

labels = ["ICs inference", "No ICs inference"]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:2],
        bbox_to_anchor=(0.55, 1.2),
        ncol=3,
        fancybox=true,)
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)

fig.savefig("figure3_temp.pdf", dpi = 300, bbox_inches="tight")