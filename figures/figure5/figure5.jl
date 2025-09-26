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
import HybridDynamicModelExperiments: HybridGrowthRateModel, VaryingGrowthRateModel, Model3SP, LogMSELoss, growth_rate_resource, water_availability, split_data, generate_noisy_data
import OrdinaryDiffEqTsit5: Tsit5
using Printf
using ComponentArrays
using Lux, NNlib
using Random
using OrdinaryDiffEqTsit5: Tsit5
include("../format.jl")

const noise = 0.1
const forecast_length = 10
function generate_data(
        ::HybridGrowthRateModel; alg, abstol, reltol, tspan, tsteps, rng, kwargs...)
    p_true = (; H = [1.24, 2.5],
        q = [4.98, 0.8],
        r = [1.0, -0.4, -0.08],
        A = [1.0],
        s = [0.8])

    u0_true = [0.5, 0.8, 0.5]
    parameters = ParameterLayer(init_value = p_true)

    lux_true_model = ODEModel(
        (; parameters), VaryingGrowthRateModel(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data
end
loss_fn = LogMSELoss()
df_baseline = []
for i in 1:5
    rng = MersenneTwister(1234 + i)
    data = generate_data(HybridGrowthRateModel();
        alg = Tsit5(),
        abstol = 1e-4,
        reltol = 1e-4,
        tspan = (0.0, 15.0),
        tsteps = 0.0:0.1:15.0,
        rng)
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


result_path_hybridgrowthrate_model = "../../scripts/luxbackend/results/luxbackend_gridsearch_hybridgrowthrate_model_with_validation_8ff34fa.jld2"
df_hybridgrowthrate = load(result_path_hybridgrowthrate_model, "results")
dropmissing!(df_hybridgrowthrate, :forecast_err)

result_path_varyinggrowthrate = "../../scripts/luxbackend/results/luxbackend_varyinggrowthrate_70fa271.jld2"
df_varyinggrowthrate = load(result_path_varyinggrowthrate, "results")
dropmissing!(df_varyinggrowthrate, :forecast_err)

df_model3sp = df_varyinggrowthrate[
    (df_varyinggrowthrate.modelname .== "Model3SP") .&& (df_varyinggrowthrate.perturb .== 1.0) .&& (df_varyinggrowthrate.noise .== 0.2),
    :]
df_refmodel = df_varyinggrowthrate[
    (df_varyinggrowthrate.modelname .== "VaryingGrowthRateModel") .&& (df_varyinggrowthrate.perturb .== 1.0) .&& (df_varyinggrowthrate.noise .== 0.2),
    :]
df_hybridgrowthrate = df_hybridgrowthrate[
    (df_hybridgrowthrate.noise .== noise) .&& (df_hybridgrowthrate.perturb .== 1.0), :]

# Calculate median forecast_error for df_hybridgrowthrate
df_hybridgrowthrate = DataFrames.transform(
    groupby(df_hybridgrowthrate, [:segment_length, :lr, :infer_ics, :weight_decay, :HlSize]),
    :forecast_err => median => :median_forecast_error)

# Calculate median forecast_error for df_model3sp
df_model3sp = DataFrames.transform(groupby(df_model3sp, [:segment_length, :lr, :infer_ics]),
    :forecast_err => median => :median_forecast_error)

# df_hybridgrowthrate = df_hybridgrowthrate[
#     df_hybridgrowthrate.median_forecast_error .== minimum(df_hybridgrowthrate.median_forecast_error),
#     :]
sorted_errors = sort(unique(df_hybridgrowthrate.median_forecast_error))
second_min = sorted_errors[1]
df_hybridgrowthrate = df_hybridgrowthrate[df_hybridgrowthrate.median_forecast_error .== second_min, :]


df = vcat(df_baseline, df_model3sp, df_refmodel, df_hybridgrowthrate, cols = :intersect)
mydict = Dict("HybridGrowthRateModel" => "Hybrid model",
    "Model3SP" => "Null model",
    "VaryingGrowthRateModel" => "Reference model")

df[!, "modelname"] = replace(df[:, "modelname"], mydict...)
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

fig = plt.figure(figsize = (6, 4))

gs = fig.add_gridspec(2, 2, width_ratios = [1, 2], height_ratios = [1, 1])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = @py fig.add_subplot(gs[0:2, 1])
axs = [ax1, ax2, ax3]

ax = ax2
color_palette = [COLORS_BR[1], COLORS_BR[3], COLORS_BR[6], COLORS_BR[end]]
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
water_avail = collect(-1.0:0.05:1)'
p_true = (; H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0],
    s = [0.8])
ys = []
losses = []
HlSize = df_hybridgrowthrate.HlSize[1]
perturb = df_hybridgrowthrate.perturb[1]
growth_rate = Lux.Chain(Lux.Dense(1, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, HlSize, NNlib.tanh),
    Lux.Dense(HlSize, 1))
_, st = Lux.setup(Random.default_rng(), growth_rate)
# lux_model = HybridDynamicModelExperiments.init(HybridGrowthRateModel();
#                                         p_true,
#                                         HlSize,
#                                         perturb,
#                                         rng = Random.default_rng())
for r in eachrow(df_hybridgrowthrate)
    ps = r.ps.growth_rate
    rates, _ = growth_rate(water_avail, ps, st)
    push!(ys, rates)
    # push!(losses, r.loss)
end
ymed = dropdims(mean(cat(ys..., dims = 3), dims = 3), dims = 3)
ymin = dropdims(minimum(cat(ys..., dims = 3), dims = 3), dims = 3)
ymax = dropdims(maximum(cat(ys..., dims = 3), dims = 3), dims = 3)
# ymin = ymed .- 3 * ystd
# ymax = ymed .+ 3* ystd
true_rates = growth_rate_resource.(Ref(p_true), water_avail)

colors = ["tab:blue", "tab:red"]

ax.fill_between(water_avail[:],
    ymin[:], ymax[:],
    # label="Neural network",
    linestyle = "-",
    color = "tab:green",
    alpha = 0.1,
    linewidth = 0.3)
ax.plot(water_avail[:],
    ymed[:],
    label = L"Feeding rate, $\text{NN}(\hat p, u)$",
    linestyle = "-",
    color = "tab:green",
    linewidth = 1.0,
    alpha = 1.0)
ax.plot(water_avail[:],
    true_rates[:],
    color = "tab:green",
    linestyle = "--",
    linewidth = 1.0)

ax.legend(handles=[
        Line2D([0], [0], color="tab:green", linestyle="-", label= "NN-based parametrization"),
        Line2D([0], [0], color="tab:green", linestyle="--", label= "Ground truth"),
        ],
        loc="lower center")
ax.set_title("Inferred growth rate")
ax.set_xlabel("Environmental forcing")
ax.set_ylabel("Basal growth rate")
display(fig)
# ax.set_yscale("log")

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