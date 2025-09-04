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

tsteps = range(500e0, step = 4, length = 100)
segmentsizes = floor.(Int, exp.(range(log(2), log(100), length = 6)))
nsegments = [length(tokens(tokenize(SegmentedTimeSeries(tsteps, segmentsize = s)))) for s in segmentsizes]

result_name_scaling_segmentsize = "../../scripts/scaling/results/scaling_7cb2a4a.jld2"
result_name_scaling_batchsize = "../../scripts/scaling/results/scaling_batch_87efebc.jld2"

df_scaling_segmentsize_nparams = load(result_name_scaling_segmentsize, "results")
nits = 5
df_scaling_segmentsize_nparams[!, :time] ./= nits * 1e9 # per iteration, in seconds (originally in ns)
dropmissing!(df_scaling_segmentsize_nparams)
df_scaling_segmentsize_nparams = flatten(df_scaling_segmentsize_nparams, :time)
df_scaling_segmentsize_nparams = df_scaling_segmentsize_nparams[df_scaling_segmentsize_nparams.optim_backend .== "LuxBackend", :]
df_scaling_segmentsize = df_scaling_segmentsize_nparams[df_scaling_segmentsize_nparams.modelname .== "Model3SP", :]
df_scaling_paramsize = df_scaling_segmentsize_nparams[df_scaling_segmentsize_nparams.segmentsize .== 9, :]

df_scaling_batchsize = load(result_name_scaling_batchsize, "results")
df_scaling_batchsize[!, :times] ./= 1e9 # per iteration, in seconds (originally in ns)
dropmissing!(df_scaling_batchsize)
df_scaling_batchsize = flatten(df_scaling_batchsize, :times)

fig, axs = plt.subplots(1, 3, figsize = (8,3))


# scaling
spread = 0.7
ax = axs[1]
ax.set_title("Batch size = 10, Model3SP")
ylab = "Simulation time\nper epoch (s)"
gdf_results = groupby(df_scaling_segmentsize, [:segmentsize, :infer_ics])
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :segmentsize,
        yname = :time, 
        xlab = "Segment size", 
        ylab = "", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette,
        legend=false)
ax.set_facecolor("none")
    ax.set_yscale("log")

ax = axs[0]
ax.set_title("Batch size = 10, segment size = 9")
gdf_results = groupby(df_scaling_paramsize, [:modelname, :infer_ics])
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :modelname,
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
# ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
x = sort!(unique(df_scaling_paramsize.modelname))
ax.set_xticks(1:length(x))
ax.set_xticklabels(x, rotation=45)
#     ax.set_yscale("log")


ax = axs[2]
ax.set_title("Segment size = 4, Model3SP")
gdf_results = groupby(df_scaling_batchsize, [:batchsize, :infer_ics])
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :batchsize,
        yname = :batchsize, 
        xlab = "Batch size", 
        ylab = "", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette,
        legend=false)
ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
x = sort!(unique(df_scaling_batchsize.batchsize)) .|> Int64
ax.set_xticks(1:length(x))
ax.set_xticklabels(x, rotation=45)
ax.set_facecolor("none")
fig.set_facecolor("none")

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
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)


fig.tight_layout()
display(fig)

fig.savefig("figure3bis_scaling.pdf", dpi = 300, bbox_inches="tight")