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

result_path_hybrid_growth_rate_model = "../../../scripts/inference_hybrid_growth_rate_model/results/2025-02-05/inference_hybrid_growth_rate_model.jld2"
@load joinpath(result_path_hybrid_growth_rate_model) results data_arr p_trues tsteps


fig, ax = plt.subplots(1, figsize=(4,3.5))

# -----------------------------
# ax1
# -----------------------------
filter!(row -> !isinf(row.loss), results)
results[!,"1/s"] = 1 ./ results.s

results[!, :val] = zeros(size(results,1))
for df in groupby(results, :s)
    s = Float32(df.s[1])
    idx_s = findfirst([p.s[1] == s for p in p_trues])
    data = data_arr[idx_s]
    for r in eachrow(df)

        mp = remake(r.res.infprob.m.mp; p = p_trues[idx_s])
        true_model = HybridGrowthRateModel(mp)
        
        # m = typeof(r.res.infprob.m).name.wrapper(mp)
        # infprob = InferenceProblem(m, r.res.infprob.p0; r.res.infprob.p_bij, r.res.infprob.u0_bij, r.res.infprob.loss_param_prior, r.res.infprob.loss_u0_prior, r.res.infprob.loss_likelihood)
        # res = InferenceResult(infprob, r.res.minloss, r.res.p_trained, r.res.u0s_trained, r.res.ranges, r.res.losses)

        r.val = validate(r.res, data, true_model)
    end
end

mydict = Dict("HybridGrowthRateModel" => L"\mathcal{M}_3^{\text{NN}}", 
            "Model3SP" => L"\mathcal{M}_3")

results[:,"scenario"] = replace(results[:,"model"], mydict...)

gdf_results = groupby(results, :noise)
df_to_plot = subset(gdf_results, :noise => x -> first(x) == 0.1)
dfg_model = groupby(df_to_plot, "scenario");


# -----------------------------
# ax2
# -----------------------------
# calculating parameter error
inv_s_to_plot = 0.1
df_to_plot = df_to_plot[df_to_plot[:, "1/s"] .== inv_s_to_plot, :]

idx_s = findfirst([p.s[1] .≈ 1/ inv_s_to_plot for p in p_trues])
p_true = p_trues[idx_s]
p_true = ComponentVector(H=p_true.H, q= p_true.q, r=p_true.r[2:end], A=p_true.A)

# Discarding unsuccessful inference results
filter!(row -> !isinf(row.loss), df_to_plot)
# selecting only r2, r3 and discarding p_nn
p_trained_filter_ar = []
for r in eachrow(df_to_plot)
    res = r.res
    if r.model == "HybridGrowthRateModel"
        r =res.p_trained["r"]
    else
        r =res.p_trained["r"][2:end]
    end
    p_trained_filter = ComponentVector(H=res.p_trained["H"], q=res.p_trained["q"], r=r, A=res.p_trained["A"])
    push!(p_trained_filter_ar, p_trained_filter)
end
df_to_plot[!, "p_trained_filtered"] = p_trained_filter_ar

par_err_median = []
for r in eachrow(df_to_plot)
    par_residual = abs.((r.p_trained_filtered .- p_true) ./ r.p_trained_filtered)
    _par_err_median = []
    for k in keys(par_residual)
        push!(_par_err_median, median(par_residual[k]))
    end
    push!(par_err_median, _par_err_median)
end
par_err_median = hcat(par_err_median...)
[df_to_plot[!, k] = par_err_median[i, :] for (i, k) in enumerate(keys(p_true))]

# PLOTTING
mydict = Dict("HybridGrowthRateModel" => "Hybrid model", 
            "Model3SP" => "Null model")
df_to_plot[:,"scenario"] = replace(df_to_plot[:,"model"], mydict...)
dfg_model = groupby(df_to_plot, "scenario");

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
pars = ["H", "q", "g", "A"]
ax.set_xticklabels(pars)
display(fig)

labels = [first(df.scenario) for df in dfg_model]
ax.legend(handles=[Line2D([0], 
        [0], 
        color=color_palette[i],
        linestyle = linestyles[i], 
        # linestyle="", 
        label=labels[i]) for i in 1:2])

ax.set_facecolor("none")
fig.set_facecolor("none")
fig.tight_layout()
display(fig)

fig.savefig(split(@__FILE__,".")[1]*".pdf", dpi = 300, bbox_inches = "tight")
