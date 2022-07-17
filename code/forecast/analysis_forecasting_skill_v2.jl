#=
plotting forecasting skill against noise and nb time series
with model of McKann

* Version
McKann model, where investigated parameter is x_p
-v2 : taking results from inference_exploration_V2
=#
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using MiniBatchInference
using DataFrames
using Glob, Dates
include("$(@__DIR__)/extrapolate_data.jl")

function score_forecasting_skill(pred_out, data_out, species=3)
    return cor(pred_out[species,:],data_out[species,:])^2
end

function get_df_forecast(df_results)
    timehorizon_lengths = 5:10:55

    df_forecasting_skill = DataFrame("res" => [], "x_p" => [], "noise" => [], "step" => [], "datasize" => [], "timehorizon_length" => [])

    for r in eachrow(df_results)
        for timehorizon_length in timehorizon_lengths
            push!(df_forecasting_skill, (r[["res", "x_p", "noise", "step", "datasize"]]..., timehorizon_length))
        end
    end

    # calculating forecasting skill for all the entries in df_results and for varying time horizon
    df_forecasting_skill[!,"forecasting_skill"] .= NaN
    # counter = 0
    for r in eachrow(df_forecasting_skill)
        try
            res, datasize, step, timehorizon_length = r[[:res,:datasize,:step,:timehorizon_length]]
            pred_in, pred_out, = extrapolate_data(res, datasize, step, timehorizon_length, false, false)
            data_in, data_out, = extrapolate_data(res, datasize, step, timehorizon_length, true, false)
            r.forecasting_skill = score_forecasting_skill(pred_out, data_out)
            # counter += 1
            # println(counter)
        catch e
            println("could not obtain results")
            println(e)
        end
    end
    return df_forecasting_skill
end


if false
    dir_res = "2022-07-12"
    name_scenario = "3-species-model_McKann_simple_minibatch_allsp"
    # name_scenario = "3-species-model_McKann_simple_minibatch_1sp"

    df_results = load("../inference_exploration_V2/results/$(dir_res)/$(name_scenario).jld2", "df_results")
    df_forecasting_skill = get_df_forecast(df_results)

    ################################
    # Forecasting plotting         #
    ################################

    # selecting a x_p of interest
    dfg_datasize = groupby(df_forecasting_skill, ["datasize"], sort = true)

    color_palette = ["tab:blue", "tab:orange", "tab:red"]
    lss = MiniBatchInference.lss


    fig, axs = subplots(1,2, sharey=true)

    for (j,df_datasize_i) in enumerate(dfg_datasize[1:2:3])
        ax = axs[j]
        dfg_noise = groupby(df_datasize_i, "noise", sort = true)[1:3]
        for (i,dfg_noise_i) in enumerate(dfg_noise)
            dfg_th = groupby(dfg_noise_i,:timehorizon_length, sort = true)
            x = [df.timehorizon_length[1] * df.step[1] for df in dfg_th ]
            y = [mean(df.forecasting_skill) for df in dfg_th ]
            ax.plot(x, y, label = "$(j) time series, r = $(dfg_noise_i.noise[1])", color = color_palette[i], linestyle = lss[j])
            ax.scatter(x, y, color = color_palette[i])
        end
        ax.set_xlabel("Time horizon", fontsize = 12); 
        j == 1 ? ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12) : nothing
        ax.legend()
    end
    display(fig)
end