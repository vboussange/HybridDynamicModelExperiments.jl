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

fig = plt.figure(figsize=(6, 4))

gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = @py fig.add_subplot(gs[0:2, 1])


result_path_func_resp_model = "../../scripts/inference_hybrid_functional_response_model/results/2025-02-06/inference_hybrid_functional_response_model_different_architectures.jld2"
@load joinpath(result_path_func_resp_model) results data p_true

# hybrid_model = HybridFuncRespModel(ModelParams(p = ComponentArray()))
# hybrid_model2 = HybridFuncRespModel2(ModelParams(p = ComponentArray()))
mp = remake(results.res[1].infprob.m.mp; p = p_true)
true_model = Model3SP(mp)

# calculating forecast error
results[!, :val] = zeros(size(results,1))
for r in eachrow(results)

        m = typeof(r.res.infprob.m).name.wrapper(mp)
        infprob = InferenceProblem(m, r.res.infprob.p0; r.res.infprob.p_bij, r.res.infprob.u0_bij, r.res.infprob.loss_param_prior, r.res.infprob.loss_u0_prior, r.res.infprob.loss_likelihood)
        res = InferenceResult(infprob, r.res.minloss, r.res.p_trained, r.res.u0s_trained, r.res.ranges, r.res.losses)

        r.val = validate(res, data, true_model)
end

# ## Fig1
ax = ax2
color_palette = ["tab:purple", "tab:orange"]
linestyles = ["--", "-."]
spread = 0.7 #spread of box plots
results = results[results.model .!== "HybridFuncRespModel2", :]
results.model .= replace(results.model, "HybridFuncRespModel" => "Hybrid model", "Model3SP" => "Reference model")
dfg_model = groupby(results, :model)
for (j,df_model_i) in enumerate(dfg_model)
    y = df_model_i.val
    bplot = ax.boxplot(y,
                positions = [j],
                showfliers = false,
                widths = 0.1,
                vert=true,  # vertical box alignment
                patch_artist=true,  # fill with color
                # notch = true,
                # label = "$(j) time series", 
                boxprops= pydict(Dict("alpha" => .3))
                )
    # putting the colors
    for patch in bplot["boxes"]
        patch.set_facecolor(color_palette[j])
        patch.set_edgecolor(color_palette[j])
    end
    for item in ["caps", "whiskers","medians"]
        for patch in bplot[item]
            patch.set_color(color_palette[j])
        end
    end
end
ax.set_xticklabels([df_model_i.model[1] for df_model_i in dfg_model], 
                rotation=45
                )
ax.set_ylabel("Forecast error")
display(fig)

ax = ax3

abundance_ranges = minimum(data, dims=2), maximum(data, dims=2)
abundance_array = range(abundance_ranges[1], stop=abundance_ranges[2], length=100)

ys = []
losses = []
results_hybrid_model = results[results.model .== "Hybrid model", :]
for r in eachrow(results_hybrid_model)
        res = r.res
        model = res.infprob.m
        p_trained = res.p_trained
        inferred_feeding_rates = hcat([feeding(hybrid_model, c, p_trained).nzval for c in abundance_array]...)
        inferred_feeding_rates .*= hcat(abundance_array...)[1:2, :]
        push!(ys,inferred_feeding_rates)
        push!(losses, r.loss)
end
ymed = dropdims(mean(cat(ys..., dims=3), dims=3), dims=3)
ymin = dropdims(minimum(cat(ys..., dims=3), dims=3), dims=3)
ymax = dropdims(maximum(cat(ys..., dims=3), dims=3), dims=3)
# ymin = ymed .- 3 * ystd
# ymax = ymed .+ 3* ystd
true_feeding_rates = hcat([feeding(true_model, c, p_true).nzval for c in abundance_array]...)
true_feeding_rates .*= hcat(abundance_array...)[1:2, :]

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
    Line2D([0], [0], color="gray", linestyle="--", label="Ground truth"),
    Line2D([0], [0], color="gray", linestyle="-", label="NN-based parametrization")
    ],
    loc="lower right", bbox_to_anchor=(1., 0.1))
ax.set_xlabel("Abundance")
ax.set_ylabel("Feeding rate")
ax.set_title("Inferred feeding rates")
ax.set_yscale("log")
# ax.set_xscale("log")

display(fig)

ax1.axis("off")

_let = ["A","B","C","D"]
for (i,ax) in enumerate([ax1, ax2, ax3])
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

