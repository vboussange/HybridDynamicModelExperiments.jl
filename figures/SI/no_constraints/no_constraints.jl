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

include("../../format.jl")
include("../../../src/loss_fn.jl")
include("../../../src/3sp_model.jl")
include("../../../src/hybrid_growth_rate_model.jl")
include("../../../src/hybrid_functional_response_model.jl")
include("../../../src/utils.jl")
include("../../../src/plotting.jl")

result_path_hybrid_growth_rate_model = "../../../scripts/inference_3sp_model_no_constraints/results/2025-02-05/inference_3sp_model_no_constraints.jld2"
@load joinpath(result_path_hybrid_growth_rate_model) results data p_true tsteps

fig, axs = plt.subplots(1,2, figsize=(6,3.5))

# -----------------------------
# ax1
# -----------------------------
ax = axs[0]
dfg_model = groupby(results, "scenario");

color_palette = ["tab:purple", "tab:orange"]
for (i, df) in enumerate(dfg_model)
    losses = hcat([r.losses for r in df.res]...)
    med_loss = median(losses, dims=2)
    min_loss = minimum(losses, dims=2)
    max_loss = maximum(losses, dims=2)
    ax.plot(med_loss[:], label = first(df.scenario), c = color_palette[i])


    ax.fill_between(1:length(med_loss), 
        min_loss[:], max_loss[:], 
            # label="Neural network",
            linestyle="-", 
            color = color_palette[i],
            alpha = 0.1,
            linewidth=0.3)
end
ax.legend()
ax.set_ylim(3, 1e2)
ax.set_xlim(0, 4e3)
ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_yscale("log")
display(fig)

# -----------------------------
# ax2
# -----------------------------
results[!, "p_trained_filtered"] = [r.p_trained for r in results.res]

par_err_median = []
for r in eachrow(results)
    par_residual = abs.((r.p_trained_filtered .- p_true) ./ r.p_trained_filtered)
    _par_err_median = []
    for k in keys(par_residual)
        push!(_par_err_median, median(par_residual[k]))
    end
    push!(par_err_median, _par_err_median)
end
par_err_median = hcat(par_err_median...)
[results[!, k] = par_err_median[i, :] for (i, k) in enumerate(keys(p_true))]

# PLOTTING
ax = axs[1]
color_palette = ["tab:purple", "tab:orange"]
linestyles = ["--", "-."]
spread = 0.7 #spread of box plots
pars = ["H", "q", "r", "A"]

for (j,df_model_i) in enumerate(dfg_model)
    y = []
    for k in pars
        push!(y, df_model_i[:,k])
    end
    xx = (1:length(pars)) .+ ((j -1) / length(pars) .- 0.5)*spread # we artificially shift the x values to better visualise the std 
    # ax.plot(x,err_arr,                
    #         color = color_palette[j] )
    bplot = ax.boxplot(y,
                positions = xx,
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

ax.set_ylabel("Parameter error")
# ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
ax.set_xticks(collect(1:length(pars)).-0.25)
ax.set_xticklabels(pars)
display(fig)




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

