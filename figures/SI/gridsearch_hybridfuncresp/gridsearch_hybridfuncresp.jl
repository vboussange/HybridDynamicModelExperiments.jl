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

result_name = "../../../scripts/luxbackend/results/luxbackend_hybridfuncresp_model_07d2781.jld2"

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

# only selecting min
# for each (segmentsize, weight_decay, infer_ics) combo, keep row with min forecast_err
# keys = [:segmentsize, :weight_decay, :infer_ics]
# df_filtered = combine(groupby(df, keys)) do sdf
#         sdf[argmin(sdf.forecast_err), :]
# end
# sort!(df_filtered, keys)

# selecting median
df_filtered = combine(groupby(df, keys),
        :forecast_err => (x -> median(skipmissing(x))) => :forecast_err,
        :forecast_err => (x -> std(skipmissing(x))) => :forecast_err_std,
        :med_par_err  => (x -> median(skipmissing(x))) => :med_par_err,
        :med_par_err  => (x -> std(skipmissing(x))) => :med_par_err_std,
)

# exporting tex table
select!(df_filtered, [:forecast_err, :forecast_err_std, :segmentsize, :weight_decay, :infer_ics])
sort!(df_filtered, keys)
display(df_filtered[df_filtered.forecast_err .== minimum(df_filtered.forecast_err),:])

df_pd = pytable(df_filtered)
min_error_idx = df_pd["forecast_err"].idxmin()
latex_code = df_pd.style.hide(
    axis="index"
).highlight_min(
    subset="forecast_err", 
    color="lightgreen", 
    axis=0
).format(format_column).to_latex(
    hrules=true,
    caption="Model performance comparison",
    label="tab:model_results"
) |> string
        
open("gridsearch_results.txt", "w") do io
        print(io, latex_code)
end


weight_decay = 1e-5
df_filtered = filter(row -> row.weight_decay == weight_decay, df)

classname = :infer_ics
classes = [true, false]

spread = 0.7 #spread of box plots

fig, axs = plt.subplots(2, 1, figsize = (6,6), sharex = "col", sharey = "row")
# averaging by nruns
gdf_results = groupby(df_filtered, [:segmentsize, classname])
ax = axs[0]

boxplot_byclass(gdf_results, ax;
        xname = :segmentsize,
        yname = :med_par_err, 
        xlab = "Segment size", 
        ylab = "Parameter error", 
        yscale = "linear", 
        classes, 
        classname, 
        spread, 
        color_palette,
        legend=false)
fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)

ax = axs[1]
boxplot_byclass(gdf_results, ax; 
        xname = :segmentsize,
        yname = :forecast_err, 
        xlab =  "Segment size", 
        ylab = "Forecast error", 
        yscale = "linear", 
        classes = classes, 
        classname, 
        spread, 
        color_palette, 
        legend=false)
ax.set_yscale("log")
ax.set_facecolor("none")
display(fig)

fig.set_facecolor("none")
ax.set_facecolor("none")
fig.tight_layout()
display(fig)

# Ms = [Model3SP, Model5SP, Model7SP]
# model_names = [L"\mathcal{M}_3", L"\mathcal{M}_5", L"\mathcal{M}_7"]
# @assert all([df.res[1].infprob.m isa M for (df,M) in zip(df_result_arr, Ms)])

fig.savefig("lr.pdf", dpi = 300, bbox_inches="tight")