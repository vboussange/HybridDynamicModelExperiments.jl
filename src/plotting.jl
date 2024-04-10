using PythonCall, PythonPlot
nx = pyimport("networkx")
np = pyimport("numpy")


function plot_foodweb(foodweb, pos, labs)
    g_nx = nx.DiGraph(np.array(adjacency_matrix(foodweb)))

    # shifting indices
    labs = Dict(keys(labs) .- 1 .=> values(labs))

    fig, ax = subplots(1)
    nx.draw(g_nx, pos, ax=ax, node_color=species_colors, node_size=1000, labels=labs)
    ax.set_facecolor("none")
    fig.set_facecolor("none")
    display(fig)
    return fig, ax
end


# plotting
using PythonPlot;
function plot_time_series(data, model)
    fig, ax = subplots()
    N = size(data, 1)

    metadata = get_metadata(model)

    for i in 1:N
        ax.plot(data[i, :], label=metadata.node_labels[i], color=metadata.species_colors[i])
    end
    # ax.set_yscale("log")
    ax.set_ylabel("Species abundance")
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("None")
    ax.set_facecolor("None")
    fig.legend()
    display(fig)
    return fig, ax
end

function plot_result_inf(res, data, model)
    fig, ax = subplots()
    tsteps = res.infprob.m.mp.kwargs[:saveat]
    metadata = get_metadata(model)
    N = size(data, 1)
    for j in 1:length(res.ranges) # looping through all segments
        for i in 1:N # looping through all species
            ax.plot(tsteps[res.ranges[j]], res.pred[j][i, :], color=metadata.species_colors[i])
        end
    end

    for i in 1:N
        ax.plot(tsteps, data[i, :], label=metadata.node_labels[i], color=metadata.species_colors[i], linestyle="--")
    end
    # ax.set_yscale("log")
    ax.set_ylabel("Species abundance")
    ax.set_xlabel("Time (days)")
    fig.set_facecolor("None")
    ax.set_facecolor("None")
    fig.legend()
    display(fig)
    return fig, ax
end

function plot_NN(res, p_true)
    water_avail = reshape(sort!(water_availability.(tsteps)), 1, :)
    model = res.infprob.m
    st = model.st

    p_nn_trained = res.p_trained.p_nn
    gr = neural_net(water_avail, p_nn_trained, st)[1]

    fig, ax = subplots(1)
    ax.plot(water_avail[:],
        gr[:],
        label="Neural network",
        linestyle="--")

    gr_true = growth_rate_resource.(Ref(p_true), water_avail)
    ax.plot(water_avail[:], gr_true[:], label="Ground truth")
    ax.set_ylim(0, 2)
    ax.set_xlim(-2, 2)
    ax.legend()
    xlabel("Water availability (normalized)")
    ylabel("Resource basal growth rate")
    display(fig)
    return fig, ax
end

function tap_results(result_name)
    df_results, data, model_true = load(result_name, "results", "data", "model_true")

    TrueParameters = model_true.mp.p

    # Discarding unsuccessful inference results
    filter!(row -> !isinf(row.loss), df_results)

    # Calculating parameter error

    # FIXME: Parameters A in current manuscript corresponds to A/K in
    # simulations, where A and K are unidentifiable. As such, for correct
    # parameter error estimation, we compute the ratio, which is identifiable.
    # For further complications, parameter K is named K₁₁ for
    # SimpleEcosystemModel3SP and K for other models 

    if occursin("3sp", result_name)
        Kname = "K₁₁"
    else
        Kname = "K"
    end
    df_results[!, "p_trained_fix"] = [ComponentVector(H=res.p_trained["H"], q=res.p_trained["q"], r=res.p_trained["r"], A=res.p_trained["A"] / res.p_trained[Kname]) for res in df_results[:, "res"]]
    TrueParameters = ComponentVector(H=TrueParameters["H"], q= TrueParameters["q"], r=TrueParameters["r"], A=TrueParameters["A"] / TrueParameters[Kname])

    df_results[!, :par_err_median] = [median(abs.((r.p_trained_fix .- TrueParameters) ./ r.p_trained_fix)) for r in eachrow(df_results)]

    # Calculating forecasting error

    # FIXME: Simulations were run with SimpleEcosystemModel3SP defined with A
    # and K parameters, but current SimpleEcosystemModel3SP has been modified
    # according to manuscript with only A parameter, so we need to assign values A/K to parameter A
    # for making correct forecasts. In the future, all models should be defined
    # according to manuscript
    if occursin("3sp", result_name)
        [r.res.p_trained["A"] .= r.p_trained_fix["A"] for r in eachrow(df_results)]
    end
    df_results[!, :val] = zeros(size(df_results, 1))
    for r in eachrow(df_results)
        try
            r.val = validate(r.res, data, model_true; length_horizon=11)
        catch e
            println(e)
            r.val = Inf
            println(r)
        end
    end
    return df_results
end



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
        display(fig)
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