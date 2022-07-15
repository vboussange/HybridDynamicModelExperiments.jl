#=
v2 : Plot AIC without ecosystem diagram
v2.2: using data from model_slection_v2

/!\ data must be first preprocessed by script "omnivory hypothesis"
=#

cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, PyCall, Printf
using DataFrames
using EcologyInformedML
using Glob
using UnPack
include("../format.jl")
include("../../code/model_selection_v2/results/analysis-omnivory.jl") # crunching the AIC 
include("../../code/model_selection_v2/results/analysis-omnivory_1sp.jl") # crunching the AIC 
include("../../code/model_selection_v2/results/analysis-omnivory_2sp.jl") # crunching the AIC 

scenario = "2sp"
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

fig, axs = plt.subplots(2, 3, sharey="row", figsize = (5.2,5))
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
for ax in axs
    ax.set_xlim(ωspan) # ω span
    ax.fill_between(ωspan, 0, 2, facecolor = "tab:grey", alpha = 0.1)
    ax.fill_between(ωspan, 4, 8, facecolor = "tab:grey", alpha = 0.3)
    ax.fill_between(ωspan, 8, ylim, facecolor = "tab:grey", alpha = 0.8)
end

axs[2].annotate("no support",[0.175, 1.])
axs[2].annotate("weak support",[0.1, 6.])
axs[1].annotate("strong support",[0.01, ylim*0.6])
gcf()

# Plotting
colors = ["tab:blue", "tab:orange"]
hyp_lab = ["Omnivory hypothesis, \n"*L"\Delta" *"AIC"*L"_{\mathcal{M}_1}", "No omnivory hypothesis, \n"*L"\Delta" *"AIC"*L"_{\mathcal{M}_2}", ]
handles = []
for (m,df_model_array) in enumerate([df_standard_model_array,df_omnivory_model_array])
    for (i,_df) in enumerate(df_model_array[idx_noise])
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

# fig.tight_layout()
gcf()

# subfigs[2].supylabel(L"\Delta"*"AIC",fontsize=15)
fig.text(0.02, 0.5, L"\Delta"*"AIC", va="center", rotation="vertical", fontsize= 10)
display(fig)

axs[2].set_ylim(-0.5,8)
axs[1].set_ylim(8,ylim)


_let = ["A","B","C","D"]
for (i,ax) in enumerate([axs[1,1],axs[1,2], axs[1,3]])
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
fig.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.05), fontsize = 10)

display(fig)

fig.savefig("figure5_$(scenario).png", dpi = 300, bbox_inches="tight")

#=
Printing numerical values
=#
using Latexify
for df in df_standard_model_array[idx_noise]
    tab_stats = latexify(df[:,[:ω,:RSS,:ΔAIC_likelihood]],env=:tabular,fmt="%.7f",latex=false) #|> String
    io = open("standard_model_AIC_r=$(df.noise[1])_$scenario.tex", "w")
    write(io,tab_stats);
    close(io)
end
for df in df_omnivory_model_array[idx_noise]
    tab_stats = latexify(df[:,[:ω,:RSS,:ΔAIC_likelihood]],env=:tabular,fmt="%.7f",latex=false) #|> String
    io = open("omnivory_model_AIC_r=$(df.noise[1])_$scenario.tex", "w")
    write(io,tab_stats);
    close(io)
end