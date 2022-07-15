#=
Investigating simulation time
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using EcologyInformedML
using DataFrames
using Glob
using Latexify

color_palette = ["tab:red", "tab:blue", "tab:green",]
linestyles = ["solid", "dashed", "dotted"]
Line2D = matplotlib.lines.Line2D #used for legend

df_results_allsp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-06-02/3-species-model_McKann_simple_minibatch_allsp.jld2", "df_results")
df_results_1sp = load("../../parameter_inference/3-species-model/inference_exploration/results/2022-06-02/3-species-model_McKann_simple_minibatch_1sp.jld2", "df_results")

# post processes
for df in [df_results_allsp,df_results_1sp]
    println(count(df.training_success), "/", size(df,1), " simulations have succeeded")
    filter!(row ->row.training_success, df)
    # need to convert p_true and p_trained
    [df[:,lab] = df[:,lab] .|> Array{Float64} for lab in ["p_true","p_trained"]]
    # converting nb_ts in int for later on
    df[!, "datasize"] = df[:, "datasize"] .|> Int
    # computing median relative error
    df[!, "rel_error"] = [mean(abs.((r.p_trained - r.p_true) / r.p_true)) for r in eachrow(df)]

    # extracting x_p
    df[!, "x_p_trained"] = [r.p_trained[2] for r in eachrow(df)]
end


time_stats_df = DataFrame("Setting" => String[],"Median simulation time" => Float64[], "Mean simulation time" => Float64[], "Std. simulation time" => Float64[])

push!(time_stats_df, ("Complete observations",  median(df_results_allsp.simtime), mean(df_results_allsp.simtime), std(df_results_allsp.simtime)))
push!(time_stats_df, ("Partial observations",   median(df_results_1sp.simtime),  mean(df_results_1sp.simtime), std(df_results_1sp.simtime)))

tab_stats = latexify(time_stats_df,env=:tabular,fmt="%.7f",latex=false) #|> String
io = open("time_stats.tex", "w")
write(io,tab_stats);
close(io)