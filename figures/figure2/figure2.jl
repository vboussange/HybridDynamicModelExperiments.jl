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
using HybridModelling
using Printf

include("../format.jl");
parallel_coordinates = pyimport("pandas.plotting").parallel_coordinates
interp1d = pyimport("scipy.interpolate").interp1d

result_name = "../../scripts/luxbackend/results/luxbackend_gridsearch_3sp_5sp_7sp_model_31bde13.jld2"

df = load(result_name, "results")
dropmissing!(df, :med_par_err)
rename!(df, "forecast_err" => "Forecast error", "med_par_err" => "Parameter error")
dims = [:segmentsize, :infer_ics, :lr]
lmodelnames = [L"\mathcal{M}_3", L"\mathcal{M}_5", L"\mathcal{M}_7"]

for perturb in unique(df.perturb), noise in unique(df.noise)
    fig, axs = plt.subplots(3, 2, figsize = (6, 5))

    for (k, metric) in enumerate(["Parameter error", "Forecast error"])
        df_filtered = df[(df.noise .== noise) .&& (df.perturb .== perturb), :]
        sm = plt.cm.ScalarMappable(cmap = CMAP_BR,
            norm = plt.Normalize(vmin = quantile(df_filtered[:, metric], 0.2), vmax = quantile(df_filtered[:, metric], 0.8)))


        for (i, modelname) in enumerate(["Model3SP", "Model5SP", "Model7SP"])
            println("Processing $modelname")
            ax = axs[i-1, k-1]
            df_filtered = df[df.modelname .== modelname, :]
            df_filtered = combine(groupby(df_filtered, dims),
                    "Forecast error" => (x -> median(skipmissing(x))) => "Forecast error",
                    "Parameter error" => (x -> median(skipmissing(x))) => "Parameter error",
                )
            # selecting median

            data = df_filtered[:, dims]

            # Normalize data to [0, 1] for each dimension
            data_norm = DataFrame()
            for col in dims
                if col in [:lr, :weight_decay, :segmentsize, :HlSize]
                    # Log scale for these columns (assuming positive values)
                    log_vals = log.(data[!, col])
                    min_val = minimum(log_vals)
                    max_val = maximum(log_vals)
                    data_norm[!, col] = (log_vals .- min_val) ./ (max_val .- min_val)
                else
                    # Linear scale for others
                    min_val = minimum(data[!, col])
                    max_val = maximum(data[!, col])
                    data_norm[!, col] = (data[!, col] .- min_val) ./ (max_val .- min_val)
                end
            end

            # Identify the top 3 combinations with lowest forecast_err
            top3_indices = sortperm(df_filtered[:, metric])[1:3]

            err = df_filtered[:, metric]
            # err = clamp.(err, quantile(err, 0.), quantile(err, 0.8))

            # Number of dimensions
            n_dims = length(dims)

            # Plot each row as a smooth line
            for i in 1:nrow(data_norm)
                y_vals = [data_norm[i, col] for col in dims]
                x_vals = 1:n_dims
                color = sm.to_rgba(err[i])
                # alpha = (err[i] - minimum(err)) / (maximum(err) - minimum(err))  # Scale alpha between 0.1 and 1.0

                # Interpolate for smooth curve
                if length(x_vals) > 1  # Ensure at least 2 points for interpolation
                    interp_func = interp1d(x_vals, y_vals, kind = "quadratic")
                    x_fine = range(1, n_dims, length = 100)  # Finer grid for smoothness
                    y_smooth = interp_func(x_fine)
                    ax.plot(x_fine, y_smooth, alpha = 0.5, color = color)  # Use a fixed color for better visibility
                else
                    # Fallback to straight line if only one point
                    ax.plot(x_vals, y_vals, alpha = 0.5, color = color)
                end
            end

            # Add axes for each dimension
            dict_ticks = Dict(:infer_ics => ["false", "true"], :lr => ["1e-3, 1e-2, 1e-1"], :segmentsize => string.(floor.(Int, exp.(range(log(2), log(100), length = 6)))))
            for j in 1:n_dims
                col = dims[j]
                # Draw vertical axis line
                ax.axvline(x=j, color="black", linewidth=1, alpha=0.5)

                # Get unique values from original data, sorted
                unique_vals = sort(unique(data[!, col]))
                min_orig = minimum(data[!, col])
                max_orig = maximum(data[!, col])

                # Get labels from dict_ticks (assume ordered)
                if haskey(dict_ticks, col)
                    if col == :infer_ics
                        labels = dict_ticks[col]  # ["false", "true"]
                        tick_norm = [0.0, 1.0]  # Fixed for boolean
                    elseif col in [:lr, :weight_decay]
                        labels = split(dict_ticks[col][1], ",")  # Split comma-separated string
                        # Normalize unique values (log scale)
                        log_min = minimum(log.(data[!, col]))
                        log_max = maximum(log.(data[!, col]))
                        tick_norm = (log.(unique_vals) .- log_min) ./ (log_max - log_min)
                    elseif col in [:segmentsize, :HlSize]
                        labels = dict_ticks[col]
                        # Normalize unique values (log scale)
                        log_min = minimum(log.(data[!, col]))
                        log_max = maximum(log.(data[!, col]))
                        tick_norm = (log.(unique_vals) .- log_min) ./ (log_max - log_min)
                    end

                    # Add ticks and labels (use min of length to avoid index errors)
                    for k in 1:min(length(unique_vals), length(labels))
                        y_pos = tick_norm[k]
                        ax.plot([j-0.05, j+0.05], [y_pos, y_pos], color="black", linewidth=1)  # Small tick line
                        t = ax.text(j-0.35, y_pos, labels[k], ha="left", va="center", fontsize=8)
                        t.set_bbox(Dict("facecolor" => "white", "alpha" => 0.7, "edgecolor" => "white", "boxstyle" => "round,pad=0.3"))
                    end
                end
            end

            # Set x-ticks and labels
            ax.set_xticks(1:n_dims)
            names = replace(dims, :segmentsize => "Segment length\n"*L"S", :infer_ics => "IC\nestimation", :lr => "Learning rate\n"*L"\gamma")
            ax.set_xticklabels([string(name) for name in names])
            # ax.set_title("Parallel Coordinates Plot Colored by Log Forecast Error")
            ax.yaxis.set_visible(false)
            ax.spines["left"].set_visible(false)
            ax.spines["right"].set_visible(false)
            ax.spines["top"].set_visible(false)
            ax.spines["bottom"].set_visible(false)

            # Add colorbar
            if modelname == "Model5SP"
                sm.set_array(err)
                # place colorbar in figure coordinates so it doesn't resize subplots
                # shrink the colorbar by creating a smaller cax and reducing font/line widths
                x0 = 0.10 + (k-1)*0.47
                y0 = -0.01
                width = 0.38
                height = 0.012  # make this smaller to visually reduce the colorbar thickness
                cax = fig.add_axes([x0, y0, width, height])
                cbar = fig.colorbar(sm, cax = cax, orientation = "horizontal")
                cbar.ax.tick_params(labelsize = 7, pad = 2)
                cbar.set_label(metric, fontsize = 8)
                cbar.ax.xaxis.set_label_position("bottom")
                cbar.outline.set_linewidth(0.5)
            end
            if k == 1
                ax.text(-0.2, 0.5, lmodelnames[i],
                    transform = ax.transAxes, ha = "center", va = "center", fontsize = 14)
            end
        end
    end

    # fig.supxlabel("Hyperparameters", fontsize = 12)
    fig.tight_layout()
    display(fig)
    fig.savefig("figure2_perturb=$(perturb)_noise=$(noise).pdf", dpi = 300, bbox_inches="tight")
end