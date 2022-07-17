#=
Plotting figure figure for section 1 of the perfect-model setting
(Figure 2)

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
-v2 : using boxplot
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using MiniBatchInference
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

dfg_datasize_allsp = groupby(df_results_allsp, ["datasize"], sort=true)
dfg_datasize_1sp = groupby(df_results_1sp, ["datasize"], sort=true)

idx_datasize = 4 # which datasize to be considered


#############################
####### plots ###############
#############################
fig, axs = subplots(2,2, 
                    figsize=(8,8), 
                    # sharey="row",
                    )
straight_line = [0.06, 0.23]

# plotting figure A (correlation ALL SPECIES vs 1 species)
ax = axs[1,1]
# ax.set_title("All species monitored\n", fontsize = 14)
ax.plot(straight_line, straight_line, color = "tab:grey", linestyle = "--")
ax.set_xlabel("True parameter, "*L"\widetilde{x_p}", fontsize = 12); ax.set_ylabel("Estimated parameter, "*L"\hat{x_p}", fontsize = 12)

labels = ["Complete obs.\n", "Partial obs.\n"]
for (i,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
    _df = groupby(df_datasize_i, "noise", sort = true)[1]
    r2 = cor(_df.x_p, _df.x_p_trained)^2
    ax.scatter(_df.x_p, 
                _df.x_p_trained,
                color = color_palette[i], 
                s = 20,
                label = labels[i]*
                    L"""R^2 = %$(@sprintf("%1.2f",r2))""")
end
ax.legend()
display(fig)

# plotting figure B
ax = axs[1,2]
labels = ["Complete obs.", "Partial obs."]
# for (j,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
#     dfg_noise_i = groupby(df_datasize_i,"noise", sort = true)
#     noise_arr = []
#     r2_arr = []
#     for (i,df_results) in enumerate(dfg_noise_i)
#         r2 = cor(df_results.x_p, df_results.x_p_trained)^2
#         push!(r2_arr,r2); push!(noise_arr,df_results.noise[1])
#     end
#     ax.plot(noise_arr,r2_arr, label = labels[j], color = color_palette[j] )
#     ax.scatter(noise_arr, r2_arr, color = color_palette[j] )
# end
# ax.legend()
# ax.set_ylabel(L"R^2")
# ax.set_ylim(0,1)
# ax.set_xlabel("Noise level, "*L"r", fontsize = 12)
# display(fig)

for (j,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
    dfg_noise_i = groupby(df_datasize_i,"noise", sort = true)
    noise_arr = []
    err_arr = []
    for (i,df_results) in enumerate(dfg_noise_i)
        push!(err_arr, df_results.rel_error); push!(noise_arr, df_results.noise[1])
    end
    x = (1:length(dfg_noise_i)) .+ (j -1) / length(dfg_noise_i) .- 0.5 # we artificially shift the x values to better visualise the std 
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
ax.set_xlabel("Noise level, "*L"r", fontsize = 12)
noise_arr = sort!(unique(df_results_1sp.noise))
ax.set_xticks(1:length(noise_arr))
ax.set_xticklabels(noise_arr)
ax.legend(handles=[Line2D([0], 
        [0], 
        color=color_palette[i], 
        # linestyle="", 
        label=labels[i]) for i in 1:2])

display(fig)
############
# Forecasts 
############

df_forecasting_skill_allsp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_allsp.jld2", "df_forecasting_skill")
df_forecasting_skill_1sp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_1sp.jld2", "df_forecasting_skill")

# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (k,df_fsk) in enumerate([df_forecasting_skill_allsp, df_forecasting_skill_1sp])
    df_datasize_i = groupby(df_fsk, ["datasize"], sort = true)[idx_datasize]
    ax = axs[2,k]
    dfg_noise = groupby(df_datasize_i, "noise", sort = true)[1:3]
    thls = [] # initialised here to be accessed at this scope

    for (i,_dfg) in enumerate(dfg_noise)
        dfg_th = groupby(_dfg,:timehorizon_length, sort = true)

        thls = []
        pred_arr = []
        for df in dfg_th
            push!(pred_arr,df.forecasting_skill); push!(thls,df.timehorizon_length[1] * df.step[1])
        end
        x = (1:length(dfg_th)) .+ (i -1) / length(dfg_th) .- 0.5 # we artificially shift the x values to better visualise the std 
        bplot = ax.boxplot(pred_arr,
            positions = x,
            showfliers = false,
            widths = 0.1,
            vert=true,  # vertical box alignment
            patch_artist=true,  # fill with color
            # notch = true,
            # label = "$(j) time series", 
        )
        ax.plot(x, median.(pred_arr), color=color_palette[i])

        # putting the colors
        for patch in bplot["boxes"]
            patch.set_facecolor(color_palette[i])
            patch.set_edgecolor(color_palette[i])
        end
        for item in ["caps", "whiskers","medians"]
            for patch in bplot[item]
                patch.set_color(color_palette[i])
            end
        end
    end
    ax.set_xlabel("Time horizon of the forecast", fontsize = 12); 
    ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12)
    # k == 1 ? ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12) : nothing
    # ax.set_ylim(0,1.1)
    ax.set_xticks(1:length(thls))
    ax.set_xticklabels(thls)
    # ax.set_yscale("log")
    # ax.legend()
end

display(fig)

axs[2,2].legend(handles=[Line2D([0], 
                                [0], 
                                color=color_palette[i], 
                                # linestyle="", 
                                label=L"r = %$(r)") for (i,r) in enumerate(sort!(unique(df_forecasting_skill_allsp.noise))[1:3])])
axs[2,1].set_title("Complete observations", fontsize = 12)
axs[2,2].set_title("Partial observations", fontsize = 12)

axs[2,1].set_ylabel("Forecast skill, "*L"\rho", fontsize = 12)
fig.supxlabel("Time horizon of the forecast", fontsize = 12)
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
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]


fig.tight_layout()
display(fig)

fig.savefig("perfect_setting_noise_boxplot.png", dpi = 300)
