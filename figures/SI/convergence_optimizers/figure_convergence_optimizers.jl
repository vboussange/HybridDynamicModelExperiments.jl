cd(@__DIR__)

using FileIO, JLD2
using DataFrames
using PythonCall 
matplotlib = pyimport("matplotlib")
plt = pyimport("matplotlib.pyplot")

include("../../../src/3sp_model.jl")
include("../../../src/loss_fn.jl")
include("../../../src/utils.jl")
include("../../../src/plotting.jl")

result_path = "../../../scripts/convergence_vs_optimizers/results/2025-01-31/convergence_vs_optimizers.jld2"
results_df, epochs = load(result_path, "results", "epochs")

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=true)
group_sizes = unique(results_df.group_size)
for (i, gsize) in enumerate(group_sizes)
    subset = filter(row -> row.group_size == gsize, eachrow(results_df))
    for row in subset
        ps = row.ps
        fill_count = epochs[1] - length(ps) + 1
        if fill_count > 0
            append!(ps, fill(ps[end], fill_count))
        end
        perr_all = [median(abs.((p_true .- p) ./ p_true)) for p in ps]
        axs[i-1].plot(1:epochs[1]+1, perr_all, label= i == 1 ? row.optim_name : nothing)
    end
    axs[i-1].set_title("Segment size: $gsize")
    axs[i-1].set_xlabel("Iteration")
    axs[i-1].set_ylabel("Parameter error")
    # axs[i-1].set_yscale("log")

    i == 1 && axs[i-1].legend(loc="upper right")
end
display(fig)
fig.savefig("convergence_vs_optimizers.pdf", bbox_inches="tight", dpi=300)