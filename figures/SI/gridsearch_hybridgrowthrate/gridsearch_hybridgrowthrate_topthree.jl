#=
Generating figure 3 of main manuscript. 

To get full figure, you need to compile `fig_3_tex/figure_3.tex`, which overlays
graphical illustration of foodwebs on top of the figure produced by this script.
=#
cd(@__DIR__)
using UnPack
using Statistics
using JLD2
using Distributions
using DataFrames
using Dates
using HybridDynamicModels
using HybridModellingExperiments: boxplot_byclass
using Printf

include("../../format.jl");

result_name = "../../../scripts/luxbackend/results/luxbackend_gridsearch_hybridgrowthrate_model_31bde13.jld2"

df = load(result_name, "results")
dropmissing!(df, :med_par_err)


function format_column(val)
    if isa(val, Bool)
        return val ? "Yes" : "No"
    elseif isa(val, Float64)
        # Use scientific notation for very small values
        if val != 0.0 && abs(val) < 1e-3
            return Printf.@sprintf("%.3e", val)
        else
            return Printf.@sprintf("%.3f", val)
        end
    else
        return string(val)
    end
end


@pyexec"""
def highlight_top_three_errors(df, column):
    # Get indices of top 3 minimum values
    top_three_idx = df[column].nsmallest(3).index
    
    def apply_highlighting(row):
        if row.name in top_three_idx:
            # Different colors for 1st, 2nd, 3rd place using CSS
            if row.name == top_three_idx[0]:  # 1st place
                return ['background-color: #90EE90'] * len(row)  # Light green
            elif row.name == top_three_idx[1]:  # 2nd place
                return ['background-color: #FFFACD'] * len(row)  # Lemon chiffon
            else:  # 3rd place
                return ['background-color: #FFDAB9'] * len(row)  # Peach puff
        return [''] * len(row)
    
    return apply_highlighting
""" => highlight_top_three_errors

# only selecting min
# for each (segmentsize, weight_decay, infer_ics) combo, keep row with min forecast_err
# keys = [:segmentsize, :weight_decay, :infer_ics]
# df_filtered = combine(groupby(df, keys)) do sdf
#         sdf[argmin(sdf.forecast_err), :]
# end
# sort!(df_filtered, keys)

# selecting median
df_keys = [:segmentsize, :HlSize, :weight_decay, :infer_ics, :noise, :lr]
df_filtered = combine(groupby(df, df_keys),
        :forecast_err => (x -> median(skipmissing(x))) => :forecast_err,
        :forecast_err => (x -> std(skipmissing(x))) => :forecast_err_std,
        :med_par_err  => (x -> median(skipmissing(x))) => :med_par_err,
        :med_par_err  => (x -> std(skipmissing(x))) => :med_par_err_std,
)

# exporting tex table
df_keys_export = vcat([:forecast_err, :forecast_err_std], df_keys)
select!(df_filtered, df_keys_export)

df_filtered = df_filtered[df_filtered.noise .== 0.2, :]
println("Best model configuration overall:")
display(df_filtered[df_filtered.forecast_err .== minimum(df_filtered.forecast_err),:])

## grid search over HlSize, segmentsize, infer_ics
df_filtered1 = df_filtered[(df_filtered.lr .== 0.001) .&& (df_filtered.weight_decay .== 1e-5), :]

sort!(df_filtered1, df_keys)
println("Best model configuration:")
display(df_filtered1[df_filtered1.forecast_err .== minimum(df_filtered1.forecast_err),:])

df_pd = pytable(df_filtered1)

latex_code = df_pd.style.hide(
    axis="index"
).apply(
    highlight_top_three_errors(df_pd, "forecast_err"),
    axis=1
).format(format_column).to_latex(
    hrules=true,
    caption="Model performance with top 3 highlighted",
    label="tab:model_results"
) |> string
        
open("gridsearch_results_hlsize_segmentsize_infer_ics.txt", "w") do io
        print(io, latex_code)
end

## grid search over weightdecay, learning rate
df_filtered2 = df_filtered[(df_filtered.HlSize .== 16) .&& (df_filtered.segmentsize .== 4) .&& (df_filtered.infer_ics .== true), :]

sort!(df_filtered2, df_keys)
println("Best model configuration:")
display(df_filtered2[df_filtered2.forecast_err .== minimum(df_filtered2.forecast_err),:])

df_pd = pytable(df_filtered2)

latex_code = df_pd.style.hide(
    axis="index"
).apply(
    highlight_top_three_errors(df_pd, "forecast_err"),
    axis=1
).format(format_column).to_latex(
    hrules=true,
    caption="Model performance with top 3 highlighted",
    label="tab:model_results"
) |> string
        
open("gridsearch_results_lr_weight_decay.txt", "w") do io
        print(io, latex_code)
end