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
using HybridDynamicModels

include("../format.jl")

tsteps = range(500e0, step = 4, length = 100)
segment_lengths = floor.(Int, exp.(range(log(2), log(100), length = 6)))
nsegments = [length(tokens(tokenize(SegmentedTimeSeries(tsteps, segment_length = s)))) for s in segment_lengths]

result_name_scaling_segment_length = "../../scripts/scaling/results/scaling_segmentsize_e75afd6.jld2"
result_name_scaling_batchsize = "../../scripts/scaling/results/scaling_batchsize_6284b1e.jld2"

df_scaling_segment_length_nparams = load(result_name_scaling_segment_length, "results")
df_scaling_segment_length_nparams[!, :time] ./= 1e9 # per iteration, in seconds (originally in ns)
dropmissing!(df_scaling_segment_length_nparams)
df_scaling_segment_length_nparams = flatten(df_scaling_segment_length_nparams, :time)
df_scaling_segment_length_nparams = df_scaling_segment_length_nparams[df_scaling_segment_length_nparams.optim_backend .== "LuxBackend", :] # legacy name
df_scaling_segment_length = df_scaling_segment_length_nparams[df_scaling_segment_length_nparams.modelname .== "Model3SP", :]
df_scaling_paramsize = df_scaling_segment_length_nparams[df_scaling_segment_length_nparams.segmentsize .== 9, :] # legacy name

df_scaling_batchsize = load(result_name_scaling_batchsize, "results")
df_scaling_batchsize[!, :times] ./= 1e9 # per iteration, in seconds (originally in ns)
dropmissing!(df_scaling_batchsize)
df_scaling_batchsize = flatten(df_scaling_batchsize, :times)

df_scaling_paramsize[!, "modelname"] = replace(df_scaling_paramsize.modelname, "Model3SP" => L"\mathcal{M}_3", "Model5SP" => L"\mathcal{M}_5", "Model7SP" => L"\mathcal{M}_7")

fig, axs = plt.subplots(1, 3, figsize = (8,3))
ylab = "Simulation time\nper epoch (s)"
spread = 0.7

ax = axs[2]
ax.set_title("Batch size = 10, segment length S = 9")
gdf_results = groupby(df_scaling_paramsize, [:modelname, :infer_ics])
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :modelname,
        yname = :time, 
        xlab = "Model", 
        ylab = "", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette= COLORS_BR[[1, length(COLORS_BR)]],
        legend=false,
        link=true)
ax.set_facecolor("none")
# ax.set_ylabel(ylab)
# ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
x = sort!(unique(df_scaling_paramsize.modelname))
ax.set_xticks(1:length(x))
ax.set_xticklabels(x)
#     ax.set_yscale("log")

# scaling
ax = axs[0]
ax.set_title("Batch size \$b = 10\$, "*L"\mathcal{M}_3")
gdf_results = groupby(df_scaling_segment_length, [:segmentsize, :infer_ics]) # legacy name
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :segmentsize,
        yname = :time, 
        xlab = "Segment length \$S\$", 
        ylab, 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette = COLORS_BR[[1, length(COLORS_BR)]],
        legend=false,
        link=true)
ax.set_facecolor("none")
ax.set_yscale("log")

ax = axs[1]
ax.set_title("Segment length S = 4, "*L"\mathcal{M}_3")
gdf_results = groupby(df_scaling_batchsize, [:batchsize, :infer_ics])
# y = [df.time for df in gdf]
boxplot_byclass(gdf_results, ax; 
        xname = :batchsize,
        yname = :times, 
        xlab = "Batch size \$b\$", 
        ylab = "", 
        yscale = "linear", 
        classes = [true, false], 
        classname = :infer_ics, 
        spread, 
        color_palette = COLORS_BR[[1, length(COLORS_BR)]],
        legend=false,
        link=true)
# ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
x = sort!(unique(df_scaling_batchsize.batchsize)) .|> Int64
ax.set_xticks(1:length(x))
ax.set_xticklabels(x, rotation=45)
ax.set_facecolor("none")
fig.set_facecolor("none")

labels = ["ICs estimated", "ICs not estimated"]
fig.legend(loc="upper center",
        handles=[Line2D([0], 
                        [0], 
                        color=COLORS_BR[[1, length(COLORS_BR)]][i],
                        # linestyle="", 
                        label=labels[i]) for i in 1:2],
        bbox_to_anchor=(0.55, 1.1),
        ncol=3,
        fancybox=true,)
# [axs[0,i-1].set_title(model_names[i], fontsize=20) for i in 1:length(df_result_arr)]
display(fig)


fig.tight_layout()
display(fig)

fig.savefig("figure3.pdf", dpi = 300, bbox_inches="tight")