#=
Plotting figure for section 2 of the perfect-model setting
(Figure 2)

* Version
- as of 25-05-2022, both forecast and parameter estimation are investigated
- rel_error: plotting abs_error instead of R2

Problem : yerr not working as expected
=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using MiniBatchInference
using DataFrames
using Glob
color_palette = ["tab:red", "tab:orange", "tab:blue", "tab:green",]

df_results_allsp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-04-18-bis/3-species-model_McKann_simple_minibatch_allsp.jld2", "df_results")
df_results_1sp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-04-18-bis/3-species-model_McKann_simple_minibatch_1sp.jld2", "df_results")

# calculating relative error
for df in [df_results_allsp,df_results_1sp]
    df[!, "rel_error"] = [median(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]
end

dfg_datasize_allsp = groupby(df_results_allsp, ["datasize"], sort=true)
dfg_datasize_1sp = groupby(df_results_1sp, ["datasize"], sort=true)

idx_datasize = [1,2,3,4] # which datasize to be considered
#############################
####### plots ###############
#############################
fig, axs = subplots(2,2, 
                    figsize=(8,8), 
                    # sharey="row",
                    )

# plotting figure A and B
labels = ["Complete obs.", "Partial obs."]

for (i,dfg) in enumerate([dfg_datasize_allsp,dfg_datasize_1sp])
    ax = axs[1,i]
    for (j,df_datasize_i) in enumerate(dfg[idx_datasize])
        dfg_noise_i = groupby(df_datasize_i,"noise", sort = true)
        noise_arr = []
        err_arr = []
        std_err_arr = []
        for (i,df_results) in enumerate(dfg_noise_i)
            err = median(df_results.rel_error)
            std_err = [quantile(df_results.rel_error, 1/4), quantile(df_results.rel_error, 3/4)]
            push!(err_arr,err);push!(std_err_arr,std_err); push!(noise_arr,df_results.noise[1])
        end
        x = noise_arr .+ j * 0.02 .- 0.05 # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        ax.errorbar(x, 
                    err_arr, 
                    yerr = hcat(std_err_arr...) .- err_arr', 
                    color = color_palette[j], 
                    fmt = "o", 
                    capsize = 5.,
                    label = "$(j) time series", )
    end
    # ax.set_ylabel(L"(\hat{p} - \Tilde{p})/\Tilde{p}")
    # ax.set_ylim(-0.05,1.1)
    ax.set_xlabel("Noise level, "*L"r", fontsize = 12)
    ax.set_xticks(noise_arr)
    # ax.set_yscale("log")
end
axs[1,1].legend()
display(fig)

############
# Forecasts 
############

df_forecasting_skill_allsp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_allsp.jld2", "df_forecasting_skill")
df_forecasting_skill_1sp = load("../../parameter_inference/3-species-model/forecast/results/df_forecasting_skill_3-species-model_McKann_simple_minibatch_1sp.jld2", "df_forecasting_skill")

# plotting figure C (# All SPECIES) and D (1 SPECIES)
for (k,df) in enumerate([df_forecasting_skill_allsp,df_forecasting_skill_1sp])
    dfg = groupby(df, ["datasize"], sort = true)[idx_datasize]
    ax = axs[2,k]
    for (i,_dfg) in enumerate(dfg)
        dfg_noise_i = groupby(_dfg, "noise", sort = true)[1]
        dfg_th = groupby(dfg_noise_i,:timehorizon_length, sort = true)

        thls = []
        pred_arr = []
        std_pred_arr = []
        for df in dfg_th
            pred = median(df.forecasting_skill)
            std_pred = [quantile(df.forecasting_skill, 1/4), quantile(df.forecasting_skill, 3/4)]
            push!(pred_arr,pred);push!(std_pred_arr,std_pred); push!(thls,df.timehorizon_length[1] * df.step[1])
        end
        dx = (thls[2] - thls[1])
        x = thls .+ i * dx / length(dfg) .- dx / 2 # we artificially shift the x values to better visualise the std 
        # ax.plot(x,err_arr,                
        #         color = color_palette[j] )
        ax.errorbar(x, 
                    pred_arr, 
                    yerr = hcat(std_pred_arr...) .- pred_arr', 
                    color = color_palette[i], 
                    fmt = "o", 
                    capsize = 5.,
                    label = "$(i) time series", )
    end
    ax.set_xlabel("Time horizon of the forecast", fontsize = 12); 
    ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12)
    # k == 1 ? ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12) : nothing
    ax.set_ylim(0,1.1)
    ax.set_xticks(noise_arr)
    # ax.legend()
end

# axs[2,1].set_ylabel("Forecast skill, "*L"\rho", fontsize = 12)
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

fig.savefig("perfect_setting_multiple_TS_rel_error.png", dpi = 300)
