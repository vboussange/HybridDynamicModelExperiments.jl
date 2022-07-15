#=
Plotting figure figure for section 1 of the perfect-model setting
(Figure 2)

Outputting numerical values in a .tex table

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
- v4 : we use data generated in inference_exploration_V2
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
include("../../code/forecast/analysis_forecasting_skill_v2.jl")
include("../format.jl")
date = "2022-07-13"
version = "_V2"
savefig = true

df_results_allsp = load("../../code/inference_exploration$(version)/results/$(date)/3-species-model_McKann_simple_minibatch_allsp.jld2", "df_results")
df_results_1sp = load("../../code/inference_exploration$(version)/results/$(date)/3-species-model_McKann_simple_minibatch_2sp.jld2", "df_results")

# post processes
for df in [df_results_allsp,df_results_1sp]
    ss = size(df,1)
    println(count(df.training_success), "/", ss, " simulations have succeeded")
    filter!(row ->row.training_success, df)
    # filtering too high RSS
    # filter!(row -> (row.RSS .< 2 * sqrt(row.noise .* mean(diag(row.Σ)))), df)
    println("Kept $(size(df,1)) / $ss points")
    # need to convert p_true and p_trained
    [df[:,lab] = df[:,lab] .|> Array{Float64} for lab in ["p_true","p_trained"]]
    # converting nb_ts in int for later on
    df[!, "datasize"] = df[:, "datasize"] .|> Int
    # computing median relative error
    df[!, "rel_error"] = [median(abs.((r.p_trained .- r.p_true) ./ r.p_true)) for r in eachrow(df)]
    df[!, "rel_error_m"] = [mean(abs.((r.p_trained - r.p_true) ./ r.p_true)) for r in eachrow(df)]

    # extracting x_p
    df[!, "x_p_trained"] = [r.p_trained[2] for r in eachrow(df)]
end

df_forecasting_skill_allsp = get_df_forecast(df_results_allsp)
df_forecasting_skill_1sp = get_df_forecast(df_results_1sp)

idx_datasize = 3 # which datasize to be considered
idx_noise_A = 2

dfg_datasize_allsp = groupby(df_results_allsp, ["datasize"], sort=true)
dfg_datasize_1sp = groupby(df_results_1sp, ["datasize"], sort=true)

## Calculating metrics of interest for printing
df_to_print_perr = DataFrame(L"r" => [], "mean param. error"=>[], "var param. error"=>[], L"R^2_{x_p}"=>[], "setting" => [])
settings = ["Complete obs.", "Partial obs."]
for (i,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
    println("datasize selected is : ", df_datasize_i.datasize[1])

    _df = groupby(df_datasize_i, "noise", sort = true)
    noise_arr = []
    err_arr = []
    r2_arr = []
    for (i,df_results) in enumerate(_df)
        push!(err_arr, df_results.rel_error); push!(noise_arr, df_results.noise[1])
        push!(r2_arr, cor(df_results.x_p, df_results.x_p_trained)^2)
    end
    _df_print = DataFrame(L"r" => noise_arr, "mean param. error"=>mean.(err_arr), "var param. error"=> var.(err_arr), L"R^2_{x_p}"=>r2_arr, "setting" => settings[i])
    append!(df_to_print_perr, _df_print)
end

#############################
####### plots ###############
#############################
fig, axs = subplots(2,2, 
                    figsize=(5.2,6), 
                    # sharey="row",
                    )
fig.subplots_adjust(hspace=0.35, wspace = 0.45)
spread = 0.3 #spread of box plots

straight_line = [0.06, 0.23]

# plotting figure A (correlation ALL SPECIES vs 1 species)
ax = axs[1,1]
# ax.set_title("All species monitored\n", fontsize = 14)
ax.plot(straight_line, straight_line, color = "tab:grey", linestyle = "--")
ax.set_xlabel("True parameter, "*L"\widetilde{x_p}", ); ax.set_ylabel("Estimated parameter, "*L"\hat{x_p}", )

labels = ["Complete obs.\n", "Partial obs.\n"]
markers = ["o", "^"]

for (i,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
    println("datasize selected is : ", df_datasize_i.datasize[1])

    _df = groupby(df_datasize_i, "noise", sort = true)[idx_noise_A] |> DataFrame
    println("r for plot A is : ", _df.noise[1])

    # filtering too high x_p
    ss = size(_df, 1)
    idx_xp = 5e-2 .< _df.x_p_trained .< 0.4
    _df = _df[idx_xp,:]
    
    # discarding datapoints with RSS higher than expected
    r2 = cor(_df.x_p, _df.x_p_trained)^2
    println("Kept $(size(_df,1)) / $ss points")

    ax.scatter(_df.x_p, 
                _df.x_p_trained,
                color = color_palette[i], 
                marker = markers[i],
                # s = 20,
                label = labels[i]*
                    L"""R^2 = %$(@sprintf("%1.2f",r2))""")
end
ax.legend()
display(fig)

# plotting figure B
ax = axs[1,2]
labels = settings
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
# ax.set_xlabel("Noise level, "*L"r", )
# display(fig)

for (j,df_datasize_i) in enumerate([dfg_datasize_allsp[idx_datasize], dfg_datasize_1sp[idx_datasize]])
    dfg_noise_i = groupby(df_datasize_i,"noise", sort = true)
    noise_arr = []
    err_arr = []
    for (i,df_results) in enumerate(dfg_noise_i)
        push!(err_arr, df_results.rel_error); push!(noise_arr, df_results.noise[1])
    end
    x = (1:length(dfg_noise_i)) .+ ((j -1) / length(idx_datasize) .- 0.5)*spread # we artificially shift the x values to better visualise the std 
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
    ax.plot(x, median.(err_arr), color=color_palette[j], linestyle = linestyles[j])
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
ax.set_ylabel("Rel. param. error, "*L"|(\hat{p} - \tilde{p})/\tilde{p}|")
ax.set_yscale("log")
# ax.set_ylim(-0.05,1.1)
ax.set_xlabel("Noise level, "*L"r", )
noise_arr = sort!(unique(df_results_1sp.noise))
ax.set_xticks(1:length(noise_arr))
ax.set_xticklabels(noise_arr)
ax.legend(handles=[Line2D([0], 
        [0], 
        color=color_palette[i],
        linestyle = linestyles[i], 
        # linestyle="", 
        label=labels[i]) for i in 1:2])

display(fig)

############
# Forecasts 
############
spread = 0.5
# [ax.set_yscale("log") for ax in axs[2,1:2]]
# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (k,df_fsk) in enumerate([df_forecasting_skill_allsp, df_forecasting_skill_1sp])
    df_fsk[:,"datasize"] = df_fsk.datasize |> Vector{Int64}
    df_datasize_i = groupby(df_fsk, ["datasize"], sort = true)[idx_datasize]
    ax = axs[2,k]
    dfg_noise = groupby(df_datasize_i, "noise", sort = true)[1:3]
    thls = [] # initialised here to be accessed at this scope

    for (i,_dfg) in enumerate(dfg_noise)
        dfg_th = groupby(_dfg,:timehorizon_length, sort = true)
        thls = []
        pred_arr = []
        for df in dfg_th
            println("we have discarded ", count(isnan.(df.forecasting_skill)), "over " , length(df.forecasting_skill) , "because NaN.")

            fs = filter(!isnan,df.forecasting_skill)

            push!(pred_arr,fs); push!(thls,df.timehorizon_length[1] * df.step[1])
        end


        x = (1:length(dfg_th)) .+ ((i -1) / 3 .- 0.5)*spread # we artificially shift the x values to better visualise the std 
        bplot = ax.boxplot(pred_arr,
            positions = x,
            showfliers = false,
            widths = 0.1,
            vert=true,  # vertical box alignment
            patch_artist=true,  # fill with color
            boxprops=Dict("alpha" => .3)
            # notch = true,
            # label = "$(j) time series", 
        )
        ax.plot(x, median.(pred_arr), color=color_palette[i], linestyle = linestyles[i],)

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
    # ax.set_xlabel("Time horizon of the forecast", ); 
    ax.set_ylabel("Forecast skill, "*L"\rho")
    # k == 1 ? ax.set_ylabel("Forecast skill, "*L"\rho", ) : nothing
    # ax.set_ylim(0,1.1)
    ax.set_xticks(1:length(thls))
    ax.set_xticklabels(thls)
    # ax.set_yscale("log")
    # ax.legend()
end

display(fig)

axs[2,1].legend(handles=[Line2D([0], 
                                [0], 
                                color=color_palette[i], 
                                linestyle = linestyles[i], 
                                # linestyle="", 
                                label=L"r = %$(r)") for (i,r) in enumerate(sort!(unique(df_forecasting_skill_allsp.noise))[1:3])])
axs[2,1].set_title("Complete observations")
axs[2,2].set_title("Partial observations")

axs[2,1].set_ylabel("Forecast skill, "*L"\rho", )
fig.supxlabel("Time horizon of the forecast", y=0.04)
display(fig)

_let = ["A","C","B","D"]
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

#=
Printing dataframe 
=#
if savefig
    fig.savefig("figure3.png", dpi = 300)
    using Latexify
    tab_stats = latexify(df_to_print_perr,env=:tabular,fmt="%.7f",latex=false) #|> String
    io = open("df_to_print_perr.tex", "w")
    write(io,tab_stats);
    close(io)
end