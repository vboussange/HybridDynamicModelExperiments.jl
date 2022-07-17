cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference
using Glob
using UnPack

@unpack df_standard_model_array, df_omnivory_model_array = load("../../parameter_inference/3-species-model/model_selection/results/2022-04-18/AIC_likelihood_comparision_omnivory-hypothesis_testing_2_simple_minibatch_step_2_datasize_50.jld2", "results_to_save")
image = plt.imread("3-species-model-omnivory-matplotlib.png")

fig = plt.figure(figsize = (8,5))
subfigs = fig.subfigures(1, 2, wspace=0.1, width_ratios=[0.8, 1.5])

###############
### img #######
###############
axim = subfigs[1].add_subplot()
axim.imshow(image)
axim.set_axis_off()
gcf()


##########################
### technical plotting ###
##########################
axs = subfigs[2].subplots(2, 3, sharey="row", gridspec_kw = Dict("height_ratios" => [1,1]))

# filling areas for support
ωspan = [-0.025, 0.525]
ylim = 3000
s_marker = 500
for ax in axs
    ax.set_xlim(ωspan) # ω span
    ax.fill_between(ωspan, 0, 2, facecolor = "tab:grey", alpha = 0.1)
    ax.fill_between(ωspan, 4, 8, facecolor = "tab:grey", alpha = 0.3)
    ax.fill_between(ωspan, 8, ylim, facecolor = "tab:grey", alpha = 0.8)
end

axs[2].annotate("no support",[0.175, 1.])
axs[2].annotate("weak support",[0.1, 6.])
axs[1].annotate("strong support",[0.05, ylim*0.7])
gcf()

# Plotting
colors = ["tab:blue", "tab:orange"]
hyp_lab = ["Omnivory hypothesis", "No omnivory hypothesis", ]
handles = []
for (m,df_model_array) in enumerate([df_standard_model_array,df_omnivory_model_array])
    for (i,_df) in enumerate(df_model_array[1:3])
        axs[1,i].set_title(L"r = %$(_df.noise[1])")
        if i == 1
            hdl = axs[1,i].scatter(_df.ω, _df[:,:ΔAICc_likelihood], c  = colors[m], label = hyp_lab[m], s = s_marker )
            push!(handles, hdl)
        else
            axs[1,i].scatter(_df.ω, _df[:,:ΔAICc_likelihood], c  = colors[m], s = s_marker )
        end
        axs[1,i].set_yscale("log")
        axs[1,i].set_xticks([])

        axs[2,i].scatter(_df.ω, _df[:,:ΔAICc_likelihood], c  = colors[m], s = s_marker)
        axs[2,i].set_xticks([0., 0.2, 0.4])
    end
end
axs[2,2].set_xlabel("Strength of omnivory ("*L"\omega"*")")
axim.legend(loc = "lower center", bbox_to_anchor=(0.6, -0.2), handles = handles)
# fig.tight_layout()
gcf()

# subfigs[2].supylabel(L"\Delta"*"AIC",fontsize=15)
subfigs[2].text(-0.01, 0.5, L"\Delta"*"AIC", fontsize=15, va="center", rotation="vertical")

axs[2].set_ylim(-0.5,8)
axs[1].set_ylim(8,ylim)


_let = ["A","B","C","D"]
for (i,ax) in enumerate([axim,axs[1,1],axs[1,2], axs[1,3]])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
    )
end

fig.subplots_adjust(hspace=0.1, wspace = 0.1)
display(fig)

fig.savefig("model_selection.png", dpi = 700, bbox_inches="tight")
