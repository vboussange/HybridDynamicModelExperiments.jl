#=
Plotting figure for section 2 of the perfect-model setting
(Figure 2)

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
- rel_error: plotting abs_error instead of R2
- v2 : we plot on the xaxis the number of TS

Problem : yerr not working as expected
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using EcologyInformedML
using DataFrames
using Glob
color_palette = ["tab:red", "tab:orange", "tab:blue", "tab:green",]
Line2D = matplotlib.lines.Line2D #used for legend

df_results_allsp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-04-18-bis/3-species-model_McKann_simple_minibatch_allsp.jld2", "df_results")
df_results_1sp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-04-18-bis/3-species-model_McKann_simple_minibatch_1sp.jld2", "df_results")

# calculating relative error
for df in [df_results_allsp,df_results_1sp]
    df[!, "rel_error"] = [median(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]
end

dfg_noise_allsp = groupby(df_results_allsp, ["noise"], sort=true)
dfg_noise_1sp = groupby(df_results_1sp, ["noise"], sort=true)

idx_datasize = [1,2,3,4] # which datasize to be considered
idx_noise = [1,2,3] # which noise is considered
#############################
####### plots ###############
#############################
fig, axs = subplots(2,2, 
                    figsize=(8,8), 
                    sharey="row",
                    )

# plotting figure A and B
labels = ["Complete obs.", "Partial obs."]

for (i,dfg) in enumerate([dfg_noise_allsp,dfg_noise_1sp])
    ax = axs[1,i]
    for (j,df_noise_i) in enumerate(dfg[idx_noise])
        global dfg_datasize_i = groupby(df_noise_i,"datasize", sort = true)
        noise_arr = []
        err_arr = []
        for (i,df_results) in enumerate(dfg_datasize_i)
            push!(err_arr, df_results.rel_error); push!(noise_arr, df_results.datasize[1])
        end
        x = (1:length(dfg_datasize_i)) .+ (j -1) / length(dfg_datasize_i) .- 0.5 # we artificially shift the x values to better visualise the std 
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
                    )
        ax.plot(x, median.(err_arr), color=color_palette[j])
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
    ax.set_xticks(1:length(dfg_datasize_i))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
axs[1,1].legend(handles=[Line2D([0], 
                                [0], 
                                color=color_palette[i], 
                                # linestyle="", 
                                label="Noise level, "*L"r = %$r") for (i,r) in enumerate(sort!(unique(df_results_allsp.noise))[idx_noise])])

# fig.legend(handles=[Line2D([0], [0], color="grey", linestyle=":", label="star graph"),
#             Line2D([0], [0], color="grey", label="complete graph")],loc="upper left") #colors

display(fig)

############
# Forecasts 
############

df_forecasting_skill_allsp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_allsp.jld2", "df_forecasting_skill")
df_forecasting_skill_1sp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_1sp.jld2", "df_forecasting_skill")
idx_timehorizon = [3,4,5]
# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (i,df) in enumerate([df_forecasting_skill_allsp,df_forecasting_skill_1sp])
    dfg = groupby(df, ["timehorizon_length"], sort = true)[idx_timehorizon]

    ax = axs[2,i]
    for (j,df_noise_i) in enumerate(dfg)
        global dfg_datasize_i = groupby(df_noise_i,"datasize", sort = true)
        noise_arr = []
        err_arr = []
        for (i,df_results) in enumerate(dfg_datasize_i)
            push!(err_arr, df_results.forecasting_skill); push!(noise_arr, df_results.datasize[1])
        end
        x = (1:length(dfg_datasize_i)) .+ (j -1) / length(dfg_datasize_i) .- 0.5 # we artificially shift the x values to better visualise the std 
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
                    )
        ax.plot(x, median.(err_arr), color=color_palette[j])
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
    ax.set_xticks(1:length(dfg_datasize_i))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
axs[2,1].legend(handles=[Line2D([0], 
                                [0], 
                                color=color_palette[i], 
                                # linestyle="", 
                                label="Time horizon of the forecast: $(r)") for (i,r) in enumerate(sort!(unique(df_forecasting_skill_allsp.timehorizon_length))[idx_timehorizon])])


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
