#=
generating figure 4 of manuscript
=#

cd(@__DIR__)
using UnPack
using Statistics
using JLD2
using Distributions
using DataFrames
using Dates
using HybridDynamicModels
import HybridDynamicModelExperiments: HybridFuncRespModel, Model3SP, feeding, LogMSELoss, generate_noisy_data, split_data
import OrdinaryDiffEqTsit5: Tsit5
using Printf
using ComponentArrays
using Lux, NNlib
using Random
using HypothesisTests
include("../format.jl")

const noise = 0.1
const forecast_length = 10
const loss_fn = LogMSELoss()

function generate_data(
        ::HybridFuncRespModel; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0])

    u0_true = [0.77, 0.060, 0.945]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data, p_true
end
df_baseline = []
for i in 1:5
    rng = MersenneTwister(1234 + i)
    data = generate_data(HybridFuncRespModel();
        alg = Tsit5(),
        abstol = 1e-4,
        reltol = 1e-4,
        tspan = (0.0, 15.0),
        tsteps = 0.0:0.1:15.0,
        rng)[1]
    data_w_noise = generate_noisy_data(data, noise, rng)
    train_idx, test_idx = split_data(data, forecast_length)
    means = median(data[:, train_idx], dims = 2)
    @show means
    preds = repeat(means, 1, length(test_idx))
    @show size(preds), size(data[:, test_idx])
    forecast_err = loss_fn(data[:, test_idx], preds)
    push!(df_baseline, (; forecast_err, modelname = "Baseline"))
end
df_baseline = DataFrame(df_baseline)


result_path_func_resp_model = "../../scripts/luxbackend/results/luxbackend_gridsearch_hybridfuncresp_model_with_validation_8ff34fa.jld2"
df_hybridfuncresp = load(result_path_func_resp_model, "results")
dropmissing!(df_hybridfuncresp, :forecast_err)

result_path_model3sp = "../../scripts/luxbackend/results/luxbackend_gridsearch_3sp_5sp_7sp_model_31bde13.jld2"
df_model3sp = load(result_path_model3sp, "results")
dropmissing!(df_model3sp, :forecast_err)

df_model3sp = df_model3sp[
    (df_model3sp.modelname .== "Model3SP") .&& (df_model3sp.perturb .== 1.0) .&& (df_model3sp.noise .== 0.2),
    :]
df_hybridfuncresp = df_hybridfuncresp[
    (df_hybridfuncresp.noise .== noise) .&& (df_hybridfuncresp.perturb .== 1.0), :]

# Calculate median forecast_error for df_hybridfuncresp
df_hybridfuncresp = DataFrames.transform(
    groupby(df_hybridfuncresp, [:segment_length, :lr, :infer_ics, :weight_decay, :HlSize]),
    :forecast_err => median => :median_forecast_error)

# Calculate median forecast_error for df_model3sp
df_model3sp = DataFrames.transform(groupby(df_model3sp, [:segment_length, :lr, :infer_ics]),
    :forecast_err => median => :median_forecast_error)

df_hybridfuncresp = df_hybridfuncresp[
    df_hybridfuncresp.median_forecast_error .== minimum(df_hybridfuncresp.median_forecast_error),
    :]
df_model3sp = df_model3sp[
    df_model3sp.median_forecast_error .== minimum(df_model3sp.median_forecast_error), :]

df = vcat(df_baseline, df_model3sp, df_hybridfuncresp, cols = :intersect)
mydict = Dict("HybridFuncRespModel" => "Hybrid model",
    "Model3SP" => "Reference model")

# Perform pairwise one-sided t-tests to assess if model X performs better than model Y (i.e., lower forecast error)
groups = groupby(df, :modelname)
model_names = unique(df.modelname)
open("pairwise_test_results.txt", "w") do io
    for i in 1:length(model_names)
        for j in (i+1):length(model_names)
            group1 = groups[i].forecast_err
            group2 = groups[j].forecast_err
            # Test if group1 (model i) has lower mean than group2 (model j)
            test = UnequalVarianceTTest(group1, group2)
            pval = pvalue(test)
            mean1 = mean(group1)
            mean2 = mean(group2)
            println(io, "Testing if $(model_names[i]) (mean: $(round(mean1, digits=4))) performs similarly to $(model_names[j]) (mean: $(round(mean2, digits=4))): p-value = $pval")
        end
    end
end

df[!, "modelname"] = replace(df[:, "modelname"], mydict...)
fig = plt.figure(figsize = (6, 4))

gs = fig.add_gridspec(2, 2, width_ratios = [1, 2], height_ratios = [1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = @py fig.add_subplot(gs[0:2, 1])
axs = [ax1, ax2, ax3]

# ## Fig1
ax = ax2
color_palette = [COLORS_BR[1], COLORS_BR[5], COLORS_BR[end]]
linestyles = ["--", "-."]
spread = 0.7 #spread of box plots
dfg_model = groupby(df, :modelname)
for (j, df_model_i) in enumerate(dfg_model)
    y = df_model_i.forecast_err
    bplot = ax.boxplot(y,
        positions = [j],
        showfliers = false,
        widths = 0.1,
        vert = true,  # vertical box alignment
        patch_artist = true,  # fill with color
        # notch = true,
        # label = "$(j) time series", 
        boxprops = pydict(Dict("alpha" => 0.3))
    )
    # putting the colors
    for patch in bplot["boxes"]
        patch.set_facecolor(color_palette[j])
        patch.set_edgecolor(color_palette[j])
    end
    for item in ["caps", "whiskers", "medians"]
        for patch in bplot[item]
            patch.set_color(color_palette[j])
        end
    end
end
ax.set_xticklabels([df_model_i.modelname[1] for df_model_i in dfg_model],
    rotation = 45
)
ax.set_ylabel("Forecast error")
# ax.set_yscale("log")
display(fig)

ax = ax3
tsteps = range(500e0, step = 4, length = 111)
data, p_true = generate_data(HybridFuncRespModel();
    alg = Tsit5(), abstol = 1e-4, reltol = 1e-4,
    tsteps,
    tspan = (0e0, tsteps[end]),
    rng = Random.default_rng())
abundance_ranges = minimum(data, dims = 2), maximum(data, dims = 2)
abundance_array = range(abundance_ranges[1], stop = abundance_ranges[2], length = 100)

ys = []
losses = []
HlSize = df_hybridfuncresp.HlSize[1]
functional_response = Lux.Chain(Lux.Dense(2, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, 2))
ps, st = Lux.setup(Random.default_rng(), functional_response)
functional_response = Lux.StatefulLuxLayer{true}(functional_response, ps, st)
hybridfuncresp_model = HybridFuncRespModel()
components = (; functional_response)
for r in eachrow(df_hybridfuncresp)
    ps = r.ps
    inferred_feeding_rates = hcat([feeding(hybridfuncresp_model, components, u, ps).nzval
                                   for u in abundance_array]...)
    inferred_feeding_rates .*= hcat(abundance_array...)[1:2, :]
    push!(ys, inferred_feeding_rates)
    # push!(losses, r.loss)
end
ymed = dropdims(mean(cat(ys..., dims = 3), dims = 3), dims = 3)
ymin = dropdims(minimum(cat(ys..., dims = 3), dims = 3), dims = 3)
ymax = dropdims(maximum(cat(ys..., dims = 3), dims = 3), dims = 3)
# ymin = ymed .- 3 * ystd
# ymax = ymed .+ 3* ystd
true_feeding_rates = hcat([feeding(Model3SP(), c, p_true).nzval for c in abundance_array]...)
true_feeding_rates .*= hcat(abundance_array...)[1:2, :]

colors = ["tab:blue", "tab:red"]

for i in 1:2
    x = hcat(abundance_array...)[i, :]
    ax.fill_between(x,
        ymin[i, :], ymax[i, :],
        # label="Neural network",
        linestyle = "-",
        color = colors[i],
        alpha = 0.1,
        linewidth = 0.3)
    ax.plot(x,
        ymed[i, :],
        label = L"Feeding rate, $\text{NN}(\hat p, u)$",
        linestyle = "-",
        color = colors[i],
        linewidth = 1.0,
        alpha = 1.0)
    ax.plot(x,
        true_feeding_rates[i, :],
        color = colors[i],
        linestyle = "--",
        linewidth = 1.0)
end
ax.legend(
    handles = [
        Line2D([0], [0], color = "tab:blue", linestyle = "-", label = "Consumer"),
        Line2D([0], [0], color = "tab:red", linestyle = "-", label = "Predator"),
        Line2D([0], [0], color = "gray", linestyle = "--", label = "Reference model"),
        Line2D(
            [0], [0], color = "gray", linestyle = "-", label = "Hybrid model")
    ],
    loc = "lower right", bbox_to_anchor = (1.0, 0.1))
ax.set_xlabel("Abundance")
ax.set_ylabel("Feeding rate")
ax.set_title("Inferred feeding rates")
ax.set_yscale("log")
# ax.set_xscale("log")

display(fig)

ax1.axis("off")

_let = ["A", "B", "C", "D"]
for (i, ax) in enumerate([ax1, ax2, ax3])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize = 12,
        fontweight = "bold",
        va = "bottom",
        ha = "left",
        transform = ax.transAxes,
        zorder = 199
    )
end

[ax.set_facecolor("none") for ax in axs]
fig.set_facecolor("none")
fig.tight_layout()
display(fig)

fig.savefig(split(@__FILE__, ".")[1] * ".pdf", dpi = 300, bbox_inches = "tight")

# %%
