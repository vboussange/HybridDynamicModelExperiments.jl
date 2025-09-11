#=
Settings for pyplot
=#
using PythonCall
# using PyCall

const matplotlib = pyimport("matplotlib")
const plt = pyimport("matplotlib.pyplot")
const Line2D = matplotlib.lines.Line2D
const LinearSegmentedColormap = matplotlib.colors.LinearSegmentedColormap

rcParams = plt."rcParams"

rcParams["font.size"] = 9
rcParams["axes.titlesize"] = 10
rcParams["axes.labelsize"] = 10
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 8
rcParams["figure.titlesize"] = 10
rcParams["lines.markersize"] = 3

color_palette = ["tab:blue", "tab:red", "tab:green"]
COLORS_BR = ["#4cc9f0","#4895ef","#4361ee","#3f37c9","#3a0ca3","#480ca8","#560bad","#7209b7","#b5179e","#f72585"]
# check https://coolors.co/palettes/popular/gradient
CMAP_BR = LinearSegmentedColormap.from_list("species_richness", COLORS_BR)

function boxplot_byclass(gdf_results, ax; xname, yname, xlab, ylab, yscale="log", classes, classname, spread, color_palette, legend, link=false)
    for (j, c) in enumerate(classes)
        df = subset(gdf_results, classname => x -> first(x) == c)
        gdf = groupby(df, xname, sort=true)

        y = [df[:, yname] for df in gdf]
        N = length(classes) # number of classes
        M = length(gdf) # number of groups
        xx = (1:M) .+ (j .- (N + 1) / 2) * spread / N # we artificially shift the x values to better visualise the std 

        boxplot(ax; y, positions=xx, color=color_palette[j])

    end

    if link
        N = length(classes)
        for (j, c) in enumerate(classes)
            df = subset(gdf_results, classname => x -> first(x) == c)
            gdf = groupby(df, xname, sort=true)
            y = [df[:, yname] for df in gdf]
            M = length(gdf)
            xx = (1:M) .+ (j .- (N + 1) / 2) * spread / N

            valid_idx = findall(v -> length(v) > 0, y)
            if !isempty(valid_idx)
                xx_valid = xx[valid_idx]
                meds = [median(y[i]) for i in valid_idx]
                ax.plot(xx_valid, meds;
                    color = color_palette[j],
                    marker = "o",
                    markerfacecolor = color_palette[j],
                    markeredgecolor = color_palette[j],
                    linewidth = 1,
                    zorder = 3
                )
            end
        end
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