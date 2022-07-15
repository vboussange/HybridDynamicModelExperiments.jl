#=
Plotting figure for section 2 of the perfect-model setting
(Figure 2)

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
- rel_error: plotting abs_error instead of R2
- v6 : we replace the x axis by noise and forecast, as showing a linear increase in the number of time series 
is not so pretty
- v6.2 : only 2 plots
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
include("../../code/forecast/analysis_forecasting_independent_TS_v2.2.jl")

date = "2022-07-13"

df_results_allsp = load("../../code/inference_exploration_V2/independent_TS_v2/results/$(date)/3-species-model_McKann_simple_minibatch_allsp_independent_TS.jld2", "df_results")
df_results_1sp = load("../../code/inference_exploration_V2/independent_TS_v2/results/$(date)/3-species-model_McKann_simple_minibatch_2sp_independent_TS.jld2", "df_results")
df_forecasting_skill_allsp = get_df_forecast_independent_TS(df_results_allsp)
df_forecasting_skill_1sp = get_df_forecast_independent_TS(df_results_1sp)

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
    df[!, "rel_error"] = [median(abs.((r.p_trained .- r.p_true) ./ r.p_true)) for r in eachrow(df)]
    df[!, "rel_error_m"] = [mean(abs.((r.p_trained .- r.p_true) ./ r.p_true)) for r in eachrow(df)]
end

dfg_noise_allsp = groupby(df_results_allsp, ["noise"], sort=true)
dfg_noise_1sp = groupby(df_results_1sp, ["noise"], sort=true)

idx_noise = 2:4 # which noise is considered
idx_ts = [1,6]
#############################
####### plots ###############
#############################
fig, axs = subplots(1,2, 
                    figsize=(8,4), 
                    # sharey="row",
                    )
spread = 1 #spread of box plots

# plotting figure A and B
labels = ["Complete obs.", "Partial obs."]

for (i,df) in enumerate([df_results_allsp, df_results_1sp])
    ax = axs[1]
    dfg_ts = groupby(df,"nb_ts", sort = true)[idx_ts]
    for (j,dfg) in enumerate(dfg_ts)
        dfg_noise = groupby(dfg, ["noise"], sort=true)[idx_noise]
        global noise_arr = []
        err_arr = []

        for df_noise_i in dfg_noise
            push!(err_arr, df_noise_i.rel_error); push!(noise_arr, df_noise_i.noise[1])
        end
        x = (1:length(dfg_noise)) .+ ((j + 2*(i+1) -2) / length(idx_noise) / 2 .- 0.5)*spread # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        bplot = ax.boxplot(err_arr,
                    positions = x,
                    showfliers = false,
                    widths = 0.1,
                    vert=true,  # vertical box alignment
                    patch_artist= true,  # fill with color
                    # notch = true,
                    # label = "$(j) time series", 
                    boxprops=Dict("alpha" => .3)
                    # alpha = 0.5
                    )
        # ax.plot(x, median.(err_arr), color=color_palette[j], linestyle = linestyles[j],)
        # ax.fill_between(x, quantile.(err_arr,0.25),quantile.(err_arr,0.75), color=color_palette[j], alpha=0.2)
        # putting the colors
        for patch in bplot["boxes"]
            j==1 ? patch.set_facecolor(color_palette[i]) : patch.set_facecolor("none")
            patch.set_edgecolor(color_palette[i])
        end
        for item in ["caps", "whiskers","medians"]
            for patch in bplot[item]
                patch.set_color(color_palette[i])
            end
        end
    end
    ax.set_ylabel("rel. param. error, "*L"(\hat{p} - \tilde{p})/\tilde{p}")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Noise level, "*L"r")
    ax.set_xticks(1:length(idx_noise))
    ax.set_xticklabels(noise_arr)
    ax.set_yscale("log")
end
# axs[1,1].legend(handles=[Line2D([0], 
#                                 [0], 
#                                 color=color_palette[i], 
#                                 linestyle = linestyles[i], 
#                                 label="Noise level, "*L"r = %$r") for (i,r) in enumerate(sort!(unique(df_results_allsp.noise))[idx_noise])])

# fig.legend(handles=[Line2D([0], [0], color="grey", linestyle=":", label="star graph"),
#             Line2D([0], [0], color="grey", label="complete graph")],loc="upper left") #colors

display(fig)

############
# Forecasts 
############

idx_timehorizon = 2:4

# fixing noise to r = 0.2
for df in [df_forecasting_skill_allsp,df_forecasting_skill_1sp]
    df[!, "noise"] = df[:, "noise"] .|> Float64
    filter!(r -> r.noise ≈ 0.2, df)
    println("Only considering r = $(df.noise[1]) for forecasting")
end

# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (i,df) in enumerate([df_forecasting_skill_allsp,df_forecasting_skill_1sp])
    dfg_nb_ts = groupby(df,"nb_ts", sort = true)[idx_ts]
    ax = axs[2]
    for (j,df_th_o) in enumerate(dfg_nb_ts)
        global dfg = groupby(df_th_o, ["timehorizon_length"], sort = true)[idx_timehorizon]
        noise_arr = []
        err_arr = []
        for (i,df_results) in enumerate(dfg)
            push!(err_arr, df_results.forecasting_skill_mean); push!(noise_arr, df_results.timehorizon_length[1])
        end
        x = (1:length(dfg)) .+ ((j + 2*(i+1) -2) / length(idx_noise) / 2 .- 0.5)*spread # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        bplot = ax.boxplot(err_arr,
                    positions = x,
                    showfliers = false,
                    widths = 0.1,
                    vert=true,  # vertical box alignment
                    patch_artist= true,  # fill with color
                    # notch = true,
                    # label = "$(j) time series", 
                    boxprops=Dict("alpha" => .3)
                    )
        # ax.plot(x, median.(err_arr), color=color_palette[j], linestyle = linestyles[j],)
        # putting the colors
        for patch in bplot["boxes"]
            j==1 ? patch.set_facecolor(color_palette[i]) : patch.set_facecolor("none")
            patch.set_edgecolor(color_palette[i])
        end
        for item in ["caps", "whiskers","medians"]
            for patch in bplot[item]
                patch.set_color(color_palette[i])
            end
        end
    end
    ax.set_ylabel("Forecast skill, "*L"\rho")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Time horizon")
    ax.set_xticks(1:length(dfg))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
nb1 = matplotlib.patches.Patch(edgecolor="grey", facecolor="none", label="Nb. of time series, "*L"S=6")
nb2 = matplotlib.patches.Patch(color="grey", label="Nb. of time series, "*L"S=1", alpha=0.3)
compl = Line2D([0], 
                [0], 
                color=color_palette[1], 
                linestyle = linestyles[1], 
                label="Complete observations")
part = Line2D([0], 
                [0], 
                color=color_palette[2], 
                linestyle = linestyles[1], 
                label="Partial observations")
fig.legend(loc="upper center",ncol=2,          
            fancybox=true,
            bbox_to_anchor=(0.5, 1.15),
            handles=[compl,part,nb2,nb1])

display(fig)

_let = ["A","B"]
for (i,ax) in enumerate(axs)
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
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

fig.savefig("figure4.png", dpi = 300, bbox_inches = "tight")
