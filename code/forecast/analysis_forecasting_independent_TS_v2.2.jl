#=
plotting forecasting skill against noise and nb time series
with model of McKann

* Version
McKann model, where investigated parameter is x_p
- independent_TS: computing the forecast skill on a single TS
-v2 : taking results from inference_exploration_V2
-v2.2 : the forecast skill is calculated as the mean forecast skill for each independent TS
=#
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using MiniBatchInference
using DataFrames
using Glob, Dates
include("$(@__DIR__)/extrapolate_data.jl")
simu = true

function score_forecasting_skill(pred_out, data_out, species=3)
    return cor(pred_out[species,:],data_out[species,:])^2
end

function get_df_forecast_independent_TS(df_results)
    filter!(row ->row.training_success, df_results)
        timehorizon_lengths = 5:10:55

    df_forecasting_skill = DataFrame("res" => [], "x_p" => [], "noise" => [], "step" => [], "datasize" => [], "nb_ts" => [], "timehorizon_length" => [])

    for r in eachrow(df_results)
        for timehorizon_length in timehorizon_lengths
            push!(df_forecasting_skill, (r[["res", "x_p", "noise", "step", "datasize", "nb_ts"]]..., timehorizon_length))
        end
    end

    # calculating forecasting skill for all the entries in df_results and for varying time horizon
    df_forecasting_skill[!,"forecasting_skill_mean"] .= NaN
    counter = 0
    for r in eachrow(df_forecasting_skill)
        try
            res, datasize, step, timehorizon_length = r[[:res,:datasize,:step,:timehorizon_length]]
            ρ_s = []
            for i in 1:length(res.pred)
                # here we reconstruct simple ResultMLE to use extrapolate_data which is not defined for independent TS results
                # we set res.ranges[1] because as in version of EIML.jl < 0.2.5, the ranges were the ones internally used by EIML 
                # we set res.ranges[1] because as in version of EIML.jl < 0.2.5, the ranges were the ones internally used by EIML 
                # we set res.ranges[1] because as in version of EIML.jl < 0.2.5, the ranges were the ones internally used by EIML 
                # and not corresponding to independent TS
                res_single_TS = ResultMLE(res.minloss,res.p_trained, res.p_true, res.p_labs, res.pred[i], res.ranges[1], res.losses, res.θs)
                pred_in_arr, pred_out_arr, = extrapolate_data(res_single_TS, datasize, step, timehorizon_length, false, false)
                data_in_arr, data_out_arr, = extrapolate_data(res_single_TS, datasize, step, timehorizon_length, true, false)
                push!(ρ_s,score_forecasting_skill(pred_out_arr, data_out_arr))
            end
            r.forecasting_skill_mean = mean(ρ_s)
            counter += 1
            # println(counter)
        catch e
            println("could not obtain results")
            println(e)
        end
    end
    return df_forecasting_skill
end

################################
# Forecasting plotting         #
################################
if false
    ####################################
    dir_res = "2022-07-11"
    # name_scenario = "3-species-model_McKann_simple_minibatch_allsp_independent_TS"
    name_scenario = "3-species-model_McKann_simple_minibatch_1sp_independent_TS"
    # name_scenario = "3-species-model_McKann_simple_minibatch_2sp_independent_TS"
    ####################################
    df_results = load("../../code/inference_exploration_V2/independent_TS_v2/results/$(date)/$(name_scenario).jld2", "df_results")

    df_forecasting_skill = get_df_forecast_independent_TS(df_results) #TODO: to be completed

    # selecting a x_p of interest
    df_forecasting_skill[!,"nb_ts"] = df_forecasting_skill[:,"nb_ts"] .|> Int
    dfg_datasize = groupby(df_forecasting_skill, ["nb_ts"], sort = true)

    color_palette = ["tab:blue", "tab:orange", "tab:red"]
    lss = MiniBatchInference.lss


    fig, axs = subplots(1,2, sharey=true)

    for (j,df_datasize_i) in enumerate(dfg_datasize[2:2:4])
        ax = axs[j]
        dfg_noise = groupby(df_datasize_i, "noise", sort = true)[1:3]
        for (i,dfg_noise_i) in enumerate(dfg_noise)
            dfg_th = groupby(dfg_noise_i,:timehorizon_length, sort = true)
            x = [df.timehorizon_length[1] * df.step[1] for df in dfg_th ]
            y = [mean(df.forecasting_skill_mean) for df in dfg_th ]
            ax.plot(x, y, label = "$(j) time series, r = $(dfg_noise_i.noise[1])", color = color_palette[i], linestyle = lss[j])
            ax.scatter(x, y, color = color_palette[i])
        end
        ax.set_xlabel("Time horizon", fontsize = 12); 
    ax.set_xlabel("Time horizon", fontsize = 12); 
        ax.set_xlabel("Time horizon", fontsize = 12); 
        j == 1 ? ax.set_ylabel("Forecast skill, "*L"\rho", fontsize = 12) : nothing
        ax.legend()
    end
    display(fig)

end