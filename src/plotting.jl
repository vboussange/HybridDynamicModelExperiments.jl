#=
Utilities for plotting and processing results.
=#
using PythonCall
const matplotlib = pyimport("matplotlib")
const Line2D = matplotlib.lines.Line2D


function boxplot_byclass(gdf_results, ax; xname, yname, xlab, ylab, yscale="log", classes, classname, spread, color_palette, legend)
    for (j, c) in enumerate(classes)
        df = subset(gdf_results, classname => x -> first(x) == c)
        gdf = groupby(df, xname, sort=true)

        y = [df[:, yname] for df in gdf]
        N = length(classes) # number of classes
        M = length(gdf) # number of groups
        xx = (1:M) .+ (j .- (N + 1) / 2) * spread / N # we artificially shift the x values to better visualise the std 

        boxplot(ax; y, positions=xx, color=color_palette[j])

    end

    # %%
    df_results = vcat(gdf_results...)
    labels = ["$classname = $n" for n in classes]
    ax.set_ylabel(ylab)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlab)
    x = sort!(unique(df_results[:, xname]))
    ax.set_xticks(1:length(x))
    ax.set_xticklabels(x, rotation=45)
    if legend
        ax.legend(handles=[Line2D([0],
            [0],
            color=color_palette[i],
            # linestyle="", 
            label=labels[i]) for i in 1:length(classes)])
    end
end

function boxplot(ax; y, positions, color)
    bplot = ax.boxplot(y,
        positions=positions,
        showfliers=false,
        widths=0.1,
        vert=true,  # vertical box alignment
        patch_artist=true,  # fill with color
        boxprops=pydict(Dict("alpha" => 0.3))
    )

    for patch in bplot["boxes"]
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    end
    for item in ["caps", "whiskers", "medians"]
        for patch in bplot[item]
            patch.set_color(color)
        end
    end
end