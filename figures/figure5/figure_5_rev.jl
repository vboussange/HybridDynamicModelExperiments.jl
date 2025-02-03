#=
generating figure 4 of manuscript
=#

cd(@__DIR__)
using PythonCall 
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")
Line2D = matplotlib.lines.Line2D #used for legend
using Graphs
using LinearAlgebra
using UnPack
using OrdinaryDiffEq
using Statistics, StatsBase
using SparseArrays
using ComponentArrays
using SciMLSensitivity
using PiecewiseInference
using JLD2
using Distributions
using Bijectors
using DataFrames
using Dates
using LaTeXStrings

include("../format.jl")
include("../../src/loss_fn.jl")
include("../../src/3sp_model.jl")
include("../../src/hybrid_growth_rate_model.jl")
include("../../src/hybrid_functional_response_model.jl")
include("../../src/utils.jl")


fig, axs = plt.subplots(1,2, figsize=(6,3.5))

ax = axs[0]


result_path_func_resp_model = "../../scripts/inference_hybrid_functional_response_model/results/2025-01-31/inference_hybrid_functional_response_model.jld2"
@load joinpath(result_path_func_resp_model) results synthetic_data p_true

hybrid_model = HybridFuncRespModel(ModelParams(p = ComponentArray()))
true_model = Model3SP(ModelParams())

abundance_ranges = minimum(synthetic_data, dims=2), maximum(synthetic_data, dims=2)
abundance_array = range(abundance_ranges[1], stop=abundance_ranges[2], length=100)

ys = []
losses = []
for r in eachrow(results)
        res = r.res
        model = res.infprob.m
        p_trained = res.p_trained
        inferred_feeding_rates = hcat([feeding(hybrid_model, c, p_trained).nzval for c in abundance_array]...)
        push!(ys,inferred_feeding_rates)
        push!(losses, r.loss)
end
ymed = dropdims(mean(cat(ys..., dims=3), AnalyticWeights(exp.(.-losses)), 3), dims=3)
ystd = dropdims(std(cat(ys..., dims=3), AnalyticWeights(exp.(.-losses)), 3), dims=3)
ymin = ymed .- 3 * ystd
ymax = ymed .+ 3* ystd
true_feeding_rates = hcat([feeding(true_model, c, p_true).nzval for c in abundance_array]...)

colors = ["tab:blue", "tab:red"]

for i in 1:2
    x = hcat(abundance_array...)[i,:]
    ax.fill_between(x, 
            ymin[i,:], ymax[i,:], 
            # label="Neural network",
            linestyle="-", 
            color = colors[i],
            alpha = 0.1,
            linewidth=0.3)
    ax.plot(x, 
            ymed[i,:],
            label=L"Feeding rate, $\text{NN}(\hat p, u)$",
            linestyle="-", 
            color = colors[i],
            linewidth=1.,
            alpha = 1.,)
    ax.plot(x, 
        true_feeding_rates[i, :], 
        color = colors[i],
        linestyle="--", 
        linewidth=1.)
end
ax.legend(handles=[
    Line2D([0], [0], color="tab:blue", linestyle="-", label="Consumer"),
    Line2D([0], [0], color="tab:red", linestyle="-", label= "Predator"),
    ],
    loc="upper right")
ax.set_xlabel("Abundance")
ax.set_ylabel("Feeding rate")
ax.set_title("Inferred feeding rates")

fig.legend(handles=[Line2D([0], [0], color="gray", linestyle="--", label="Ground truth"),
                    Line2D([0], [0], color="gray", linestyle="-", label="NN-based parametrization")],
            loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=1)

display(fig)

# -----------------------------
# ax2
# -----------------------------


# # ## Fig1
# ax = axs[0]
# color_palette = ["tab:purple", "tab:orange"]
# linestyles = ["--", "-."]
# spread = 0.7 #spread of box plots
# for (j,df_model_i) in enumerate(dfg_model)
#     dfg_model_i = groupby(df_model_i,"1/s", sort = true)
#     y = []
#     for (i,results) in enumerate(dfg_model_i)
#         push!(y, results.val)
#     end
#     xx = (1:length(dfg_model_i)) .+ ((j -1) / length(dfg_model_i) .- 0.5)*spread # we artificially shift the x values to better visualise the std 
#     # ax.plot(x,err_arr,                
#     #         color = color_palette[j] )
#     bplot = ax.boxplot(y,
#                 positions = xx,
#                 showfliers = false,
#                 widths = 0.1,
#                 vert=true,  # vertical box alignment
#                 patch_artist=true,  # fill with color
#                 # notch = true,
#                 # label = "$(j) time series", 
#                 boxprops= pydict(Dict("alpha" => .3))
#                 )
#     ax.plot(xx, median.(y), color=color_palette[j], linestyle = linestyles[j])
#     # putting the colors
#     for patch in bplot["boxes"]
#         patch.set_facecolor(color_palette[j])
#         patch.set_edgecolor(color_palette[j])
#     end
#     for item in ["caps", "whiskers","medians"]
#         for patch in bplot[item]
#             patch.set_color(color_palette[j])
#         end
#     end
# end



# # %%
# labels = [first(df.scenario) for df in dfg_model]
# ax.set_ylabel("Forecast error")
# # ax.set_yscale("log")
# # ax.set_ylim(-0.05,1.1)
# ax.set_xlabel(L"1/s")
# x = sort!(unique(df_to_plot."1/s"))
# x = round.(x, digits=1)
# ax.set_xticks(collect(1:length(x)).-0.25)
# ax.set_xticklabels(x)
# ax.legend(handles=[Line2D([0], 
#         [0], 
#         color=color_palette[i],
#         linestyle = linestyles[i], 
#         # linestyle="", 
#         label=labels[i]) for i in 1:2])


result_path_hybrid_growth_rate_model = "../../scripts/inference_hybrid_growth_rate_model/results/2025-01-31/inference_hybrid_growth_rate_model.jld2"
@load joinpath(result_path_hybrid_growth_rate_model) results data_arr p_trues

mydict = Dict("HybridGrowthRateModel" => L"\mathcal{M}_3^{\text{NN}}", 
            "Model3SP" => L"\mathcal{M}_3")

results[:,"scenario"] = replace(results[:,"model"], mydict...)

gdf_results = groupby(results, :noise)
df_to_plot = subset(gdf_results, :noise => x -> first(x) == 0.1)
dfg_model = groupby(df_to_plot, "scenario");

# %%
s_to_plot = 0.8f0
df_to_plot = subset(dfg_model, :scenario => x -> first(x) == latexstring("\$\\mathcal{M}_3^{\\text{NN}}\$"))
df_to_plot = df_to_plot[df_to_plot.s .== s_to_plot, :]

# %%
water_avail = collect(-1.:0.05:1)'
ys = []
losses = []
for r in eachrow(df_to_plot)
        res = r.res
        model = res.infprob.m
        p_nn_trained = res.p_trained.p_nn
        gr = model.growth_rate(water_avail, p_nn_trained)
        push!(ys,gr)
        push!(losses, r.loss)
end
ymed = mean(vcat(ys...), dims=1, AnalyticWeights(exp.(.-losses)))
ystd = std(vcat(ys...), AnalyticWeights(exp.(.-losses)), 1)
ymin = ymed .- 3 * ystd
ymax = ymed .+ 3 * ystd

# %%
ax = axs[1]

ax.fill_between(water_avail[:], 
        ymin[:], ymax[:], 
        # label="Neural network",
        linestyle="-", 
        color = "tab:orange",
        alpha = 0.1,
        linewidth=0.3)

ax.plot(water_avail[:], 
        ymed[:], 
        linestyle="-", 
        color = "tab:orange",
        linewidth=1.,
        alpha = 1.)
idx_s = findfirst([p.s[1] == s_to_plot for p in p_trues])
p_true = p_trues[idx_s]
gr_true = growth_rate_resource.(Ref(p_true), water_avail)
ax.plot(water_avail[:], 
        gr_true[:], 
        linestyle="--", 
        linewidth=1.,
        # label="True growth rate",
        color = "tab:orange")
ax.legend(handles=[
        Line2D([0], [0], color="tab:orange", linestyle="-", label= "Primary producer"),
        ],
        loc="upper right")
ax.set_title("Inferred growth rate")
ax.set_xlabel("Environmental forcing")
ax.set_ylabel("Basal growth rate")
display(fig)
# %%

# Load the image with the extension .pdf
# image_path = "../fig_2_3-species-model-diagram/figure2b.png"
# img = mpimg.imread(image_path)

# # Display the image in the first subplot (ax1)
# axs[0].imshow(img)
# axs[0].axis("off")  # Turn off axis ticks and labels for better visualization

_let = ["A","B","C","D"]
for (i,ax) in enumerate(axs)
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
        zorder = 199
    )
end

[ax.set_facecolor("none") for ax in axs]
fig.set_facecolor("none")
fig.tight_layout()
display(fig)

fig.savefig(split(@__FILE__,".")[1]*".pdf", dpi = 300, bbox_inches = "tight")

# %%

