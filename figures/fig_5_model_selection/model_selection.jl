#=
Plotting Aikaike weights

=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using MiniBatchInference
using Glob
using UnPack
include("../format.jl")
include("../../code/model_selection_v2/results/analysis-omnivory.jl") # crunching the AIC 
include("../../code/model_selection_v2/results/analysis-omnivory_2sp.jl") # crunching the AIC 

scenario = "allsp"
date = "2022-07-14"

if scenario == "2sp"
    scenario_name = "omnivory-hypothesis_testing_v2_simple_minibatch_step_4_datasize_60_2sp" # can be "1sp" or "allsp"
    get_results = get_results_AIC_omnivory_2sp
else
    scenario_name = "omnivory-hypothesis_testing_v2_simple_minibatch_step_4_datasize_60_allsp" # can be "1sp" or "allsp"
    get_results = get_results_AIC_omnivory_allsp
end

@load "../../code/model_selection_v2/results/$(date)/$(scenario_name).jld2" df_results
df_standard_model_array, df_omnivory_model_array = get_results(df_results)

fig, axs = plt.subplots(1, 3, sharey="row", figsize = (5.2,2.5))
fig.set_facecolor("None")
[ax.set_facecolor("None") for ax in axs]


##########################
### technical plotting ###
##########################
idx_noise = 1:1:3 # noise level to be plotted


# filling areas for support
ωspan = [-0.025, 0.525]
ylim = 3000
s_marker = 10

# Plotting
colors = ["tab:blue", "tab:orange"]
hyp_lab = ["No omnivory hypothesis, \n"*L"{\mathcal{M}_1}", "Omnivory hypothesis, \n"*L"{\mathcal{M}_2}", ]
handles = []
for (m,df_model_array) in enumerate([df_standard_model_array,df_omnivory_model_array])
    for (i,_df) in enumerate(df_model_array[idx_noise])
        axs[i].set_title(L"r = %$(_df.noise[1])")
        if i == 1
            hdl = axs[i].scatter(_df.ω, _df[:,:W], c  = colors[m], label = hyp_lab[m], s = s_marker )
            push!(handles, hdl)
        else
            axs[i].scatter(_df.ω, _df[:,:W], c  = colors[m], s = s_marker )
        end
        axs[i].set_xticks([0., 0.2, 0.4])
        axs[i].set_yticks([0.:0.2:1.0;])

    end
end
axs[2].set_xlabel("Strength of omnivory, "*L"\omega")

# fig.tight_layout()
gcf()

# subfigs[2].supylabel(L"\Delta"*"AIC",fontsize=15)
fig.text(0.02, 0.5, "Model probability, "*L"w_{\mathcal{M}_i}", va="center", rotation="vertical", fontsize= 10)
display(fig)

axs[2].set_ylim(-0.2,1.2)
# axs[1].set_ylim(10,ylim)


_let = ["A","B","C","D"]
for (i,ax) in enumerate(axs)
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
        zorder = 199
    )
end

fig.subplots_adjust(hspace=0.1, wspace = 0.1)
fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize = 10)

display(fig)

fig.savefig("figure5_$(scenario).png", dpi = 300, bbox_inches="tight")

#=
Printing numerical values
=#
using Latexify
for df in df_standard_model_array[idx_noise]
    tab_stats = latexify(df[:,[:ω,:RSS,:ΔAIC_likelihood,:W]],env=:tabular,fmt="%.7f",latex=false) #|> String
    io = open("standard_model_AIC_r=$(df.noise[1])_$scenario.tex", "w")
    write(io,tab_stats);
    close(io)
end
for df in df_omnivory_model_array[idx_noise]
    tab_stats = latexify(df[:,[:ω,:RSS,:ΔAIC_likelihood,:W]],env=:tabular,fmt="%.7f",latex=false) #|> String
    io = open("omnivory_model_AIC_r=$(df.noise[1])_$scenario.tex", "w")
    write(io,tab_stats);
    close(io)
end