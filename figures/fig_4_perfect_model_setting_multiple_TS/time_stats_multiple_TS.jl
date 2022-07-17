#=
Investigating simulation time vs nb of datapoints
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using MiniBatchInference
using DataFrames
using Glob
color_palette = ["tab:red", "tab:blue", "tab:green",]
linestyles = ["solid", "dashed", "dotted"]
Line2D = matplotlib.lines.Line2D #used for legend

df_results_allsp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-06-15/3-species-model_McKann_simple_minibatch_allsp_independent_TS.jld2", "df_results")
df_results_1sp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-06-15/3-species-model_McKann_simple_minibatch_1sp_independent_TS.jld2", "df_results")

# post processes
for df in [df_results_allsp,df_results_1sp]
    println(count(df.training_success), "/", size(df,1), " simulations have succeeded")
    filter!(row ->row.training_success, df)
    # need to convert p_true and p_trained
    [df[:,lab] = df[:,lab] .|> Array{Float64} for lab in ["p_true","p_trained"]]
    # converting nb_ts in int for later on
    df[!, "nb_ts"] = df[:, "nb_ts"] .|> Int
    # computing median relative error
    df[!, "rel_error"] = [median(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]
end

dfg_noise_allsp = groupby(df_results_allsp, ["noise"], sort=true)
dfg_noise_1sp = groupby(df_results_1sp, ["noise"], sort=true)

idx_NTS = [1,2,3,4] # which datasize to be considered
idx_noise = [2] # which noise is considered
#############################
####### plots ###############
#############################
fig, axs = subplots(1,2,
                    figsize=(8,4), 
                    sharey="row",
                    )
spread = 0.5 #spread of box plots

# plotting figure A and B
labels = ["Complete obs.", "Partial obs."]

for (i,dfg) in enumerate([dfg_noise_allsp,dfg_noise_1sp])
    ax = axs[i]
    for (j,df_noise_i) in enumerate(dfg[idx_noise])
        global dfg_nb_ts = groupby(df_noise_i,"nb_ts", sort = true)
        global noise_arr = []
        time_sim = []
        for (i,df_results) in enumerate(dfg_nb_ts)
            push!(time_sim, df_results.simtime); push!(noise_arr, df_results.nb_ts[1])
        end
        x = (1:length(dfg_nb_ts)) # we artificially shift the x values to better visualise the std 
        # ax.plot(x,time_sim,                
        #         color = color_palette[j] )
        bplot = ax.boxplot(time_sim,
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
        ax.plot(x, median.(time_sim), color=color_palette[j], linestyle = linestyles[j],)
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
    ax.set_ylabel("Simulation time (s)")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Nb. of time series, "*L"S", fontsize = 12)
    ax.set_xticks(1:length(dfg_nb_ts))
    ax.set_xticklabels(noise_arr)
    # ax.set_yscale("log")
end
# axs[1,1].legend(handles=[Line2D([0], 
#                                 [0], 
#                                 color=color_palette[i], 
#                                 linestyle = linestyles[i], 
#                                 label="Noise level, "*L"r = %$r") for (i,r) in enumerate(sort!(unique(df_results_allsp.noise))[idx_noise])])

# fig.legend(handles=[Line2D([0], [0], color="grey", linestyle=":", label="star graph"),
#             Line2D([0], [0], color="grey", label="complete graph")],loc="upper left") #colors
axs[1].set_title("Complete observations\n", fontsize = 15)
axs[2].set_title("Partial observations\n", fontsize = 15)

display(fig)

_let = ["A","B"]
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

fig.savefig("time_stats_multiple_TS.png", dpi = 300)
