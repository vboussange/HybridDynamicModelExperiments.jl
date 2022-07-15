#=
Plotting figure for section 2 of the perfect-model setting
(Figure 2)

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
- rel_error: plotting abs_error instead of R2
- v5 : we use data generated in inference_exploration_V2
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using EcologyInformedML
using DataFrames
using Glob
color_palette = ["tab:red", "tab:blue", "tab:green",]
linestyles = ["solid", "dashed", "dotted"]
Line2D = matplotlib.lines.Line2D #used for legend

df_results_allsp = load("../../parameter_inference/3-species-model/inference_exploration_V2/independent_TS_v2/results/2022-07-04/3-species-model_McKann_simple_minibatch_allsp_independent_TS.jld2", "df_results")
df_results_1sp = load("../../parameter_inference/3-species-model/inference_exploration_V2/independent_TS_v2/results/2022-07-04/3-species-model_McKann_simple_minibatch_1sp_independent_TS.jld2", "df_results")

# post processes
for df in [df_results_allsp,df_results_1sp]
    println(count(df.training_success), "/", size(df,1), " simulations have succeeded")
    filter!(row ->row.training_success, df)
    filter!(row ->row.training_success, df)

    # need to convert p_true and p_trained
    [df[:,lab] = df[:,lab] .|> Array{Float64} for lab in ["p_true","p_trained"]]
    # converting nb_ts in int for later on
    df[!, "nb_ts"] = df[:, "nb_ts"] .|> Int
    # computing median relative error
    df[!, "rel_error"] = [median(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]
    df[!, "rel_error_m"] = [mean(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]
end

dfg_noise_allsp = groupby(df_results_allsp, ["noise"], sort=true)
dfg_noise_1sp = groupby(df_results_1sp, ["noise"], sort=true)

idx_noise = 2:4 # which noise is considered
idx_ts = [1,6]
#############################
####### plots ###############
#############################
fig, axs = subplots(2,2, 
                    figsize=(8,8), 
                    # sharey="row",
                    )
spread = 0.5 #spread of box plots

# plotting figure A and B
labels = ["Complete obs.", "Partial obs."]

for (i,dfg) in enumerate([dfg_noise_allsp,dfg_noise_1sp])
    ax = axs[1,i]
    for (j,df_noise_i) in enumerate(dfg[idx_noise])
        global dfg_nb_ts = groupby(df_noise_i,"nb_ts", sort = true)[idx_ts]
        global noise_arr = []
        err_arr = []
        for (i,df_results) in enumerate(dfg_nb_ts)
            push!(err_arr, df_results.rel_error_m); push!(noise_arr, df_results.nb_ts[1])
        end
        x = (1:length(dfg_nb_ts)) .+ ((j -1) / length(idx_noise) .- 0.5)*spread # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        bplot = ax.boxplot(err_arr,
                    positions = x,
                    showfliers = false,
                    widths = 0.1,
                    vert=true,  # vertical box alignment
                    patch_artist=true,  # fill with color
                    # notch = true,
                    # label = "$(j) time series", 
                    boxprops=Dict("alpha" => .3)
                    # alpha = 0.5
                    )
        ax.plot(x, median.(err_arr), color=color_palette[j], linestyle = linestyles[j],)
        # ax.fill_between(x, quantile.(err_arr,0.25),quantile.(err_arr,0.75), color=color_palette[j], alpha=0.2)
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
    ax.set_ylabel("rel. param. error, "*L"(\hat{p} - \tilde{p})/\tilde{p}")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Nb. of time series, "*L"S", fontsize = 12)
    ax.set_xticks(1:length(dfg_nb_ts))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
axs[1,1].legend(handles=[Line2D([0], 
                                [0], 
                                color=color_palette[i], 
                                linestyle = linestyles[i], 
                                label="Noise level, "*L"r = %$r") for (i,r) in enumerate(sort!(unique(df_results_allsp.noise))[idx_noise])])

# fig.legend(handles=[Line2D([0], [0], color="grey", linestyle=":", label="star graph"),
#             Line2D([0], [0], color="grey", label="complete graph")],loc="upper left") #colors

display(fig)

############
# Forecasts 
############

df_forecasting_skill_allsp = load("../../parameter_inference/3-species-model/forecast/results/v2/df_forecasting_skill_3-species-model_McKann_simple_minibatch_allsp_independent_TS.jld2", "df_forecasting_skill")
df_forecasting_skill_1sp = load("../../parameter_inference/3-species-model/forecast/results/v2/df_forecasting_skill_3-species-model_McKann_simple_minibatch_1sp_independent_TS.jld2", "df_forecasting_skill")
idx_timehorizon = 2:4

# fixing noise to r = 0.2
for df in [df_forecasting_skill_allsp,df_forecasting_skill_1sp]
    df[!, "noise"] = df[:, "noise"] .|> Float64
    filter!(r -> r.noise ≈ 0.2, df)
    println("Only considering r = $(df.noise[1]) for forecasting")
end

# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (i,df) in enumerate([df_forecasting_skill_allsp,df_forecasting_skill_1sp])
    dfg = groupby(df, ["timehorizon_length"], sort = true)[idx_timehorizon]
    ax = axs[2,i]
    for (j,df_th_o) in enumerate(dfg)
        global dfg_nb_ts = groupby(df_th_o,"nb_ts", sort = true)[idx_ts]
        noise_arr = []
        err_arr = []
        for (i,df_results) in enumerate(dfg_nb_ts)
            push!(err_arr, df_results.forecasting_skill_mean); push!(noise_arr, df_results.nb_ts[1])
        end
        x = (1:length(dfg_nb_ts)) .+ ((j -1) / length(idx_timehorizon) .- 0.5)*spread # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        bplot = ax.boxplot(err_arr,
                    positions = x,
                    showfliers = false,
                    widths = 0.1,
                    vert=true,  # vertical box alignment
                    patch_artist=true,  # fill with color
                    # notch = true,
                    # label = "$(j) time series", 
                    boxprops=Dict("alpha" => .3)
                    )
        ax.plot(x, median.(err_arr), color=color_palette[j], linestyle = linestyles[j],)
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
    ax.set_ylabel("Forecast skill, "*L"\rho")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Nb. of time series, "*L"S", fontsize = 12)
    ax.set_xticks(1:length(dfg_nb_ts))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
axs[2,1].legend(loc="lower right",
                handles=[Line2D([0], 
                        [0], 
                        color=color_palette[i], 
                        linestyle = linestyles[i], 
                        label="Time horizon: $(r)") for (i,r) in enumerate(sort!(unique(df_forecasting_skill_allsp.timehorizon_length))[idx_timehorizon])])


# axs[2,1].set_ylabel("Forecast skill, "*L"\rho", fontsize = 12)
axs[1,1].set_title("Complete observations\n", fontsize = 15)
axs[1,2].set_title("Partial observations\n", fontsize = 15)

display(fig)

_let = ["A","C","B","D"]
for (i,ax) in enumerate(axs)
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
    )
end

fig.tight_layout()
display(fig)

fig.savefig("perfect_setting_multiple_TS_rel_error_boxplot_v2.png", dpi = 300)
