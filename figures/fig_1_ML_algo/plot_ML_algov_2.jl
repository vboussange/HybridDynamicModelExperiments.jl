#=
Figure 1 in main manuscript.

----
* VERSION
v2: 4 figures, showing the splitting
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
# LATEX needed

py"""
import os
os.environ["PATH"]='/usr/local/opt/ruby/bin:/usr/local/lib/ruby/gems/3.0.0/bin:/Users/victorboussange/.gem/ruby/2.6.0/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Applications/Visual Studio Code.app/Contents/Resources/app/bin:/usr/local/anaconda3/bin:/Applications/Julia-1.0.app/Contents/Resources/julia/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin'
"""
rcParams["text.usetex"] = true
rcParams["text.latex.preamble"] = 
                                "\\usepackage{amsmath}
                                \\usepackage{DejaVuSans}
                                \\usepackage{sansmath}
                                \\sansmath
                                "
color_palette = ["tab:red", "tab:blue", "tab:green",]
linestyles = ["solid", "dashed", "dotted"]

# DATA FOR LOSSES AND TIME SERIES
res_batch = load("results/2022-06-20/batch_inference.jld2", "res") 
res_naive = load("results/2022-06-20/naive_inference.jld2", "res") 
res_naive_neigh = load("results/2022-06-20/naive_inference_neighbourhood.jld2", "res") 

ode_datas_wnoise, tsteps, ranges = load("ode_datas_wnoise.jld2", "ode_datas_wnoise", "tsteps", "ranges")
ranges = ranges[1:4]

# DATA FOR PROFILE LOSS
@load "../../code/profile_likelihood/plikelihood_datasize=200.jld2"

dim_prob = 3 #ODE dimension

masterfig = plt.figure(figsize = (5.2,5))
masterfig.set_facecolor("None")
subfigs = masterfig.subfigures(2, 2, wspace=0.1, width_ratios=[1.,1.5])
[fig.set_facecolor("None") for fig in subfigs]

##########################
### plotting profile loss ###
##########################
ax_loss = subfigs[1].subplots()
# ax.set_title("Time horizon T = $(tsteps[end] - tsteps[1])")
ax_loss.plot(x_ps, p_likelihood, label = L"L_{\mathcal{M}}(\theta)", linestyle = linestyles[1])
ax_loss.plot(x_ps, p_likelihood_EIML, label = L"L_{\mathcal{M}}^\star(\theta)", linestyle = linestyles[2])
ax_loss.axvline(p_true,  label = L"\Tilde p_2", linestyle = linestyles[3], color = "red")
# ax_loss.set_ylabel(L"L(p_2)")
ax_loss.legend(loc="best", bbox_to_anchor=(0.5, 0.45, 0.4, 0.5))
ax_loss.set_xlabel(L"p_2",y=0.2,zorder=10)

display(masterfig)


################################
# plotting loss vs iterations  #
################################
ax_its_loss = subfigs[2,1].subplots()
ax_its_loss.plot(1:length(res_batch.θs)-1, res_batch.θs[1:end-1], label = L"L_{\mathcal{M}}^\star(\theta)", linestyle = linestyles[1])
ax_its_loss.plot(1:length(res_naive.θs)-1, res_naive.θs[1:end-1], label = L"L_{\mathcal{M}}(\theta)", linestyle = linestyles[2])
ax_its_loss.plot(1:length(res_naive_neigh.θs)-1, res_naive_neigh.θs[1:end-1], label = L"L_{\mathcal{M}}(\Tilde{\theta} + \delta\theta)", linestyle = linestyles[3])
ax_its_loss.legend(loc="lower right")
ax_its_loss.set_yscale("log")
ax_its_loss.set_ylabel(L"\|\theta_m - \Tilde\theta \|")

ax_its_loss.set_xlabel("Epochs")
[ax.set_facecolor("None") for ax in [ax_its_loss,ax_loss]]
display(masterfig)


data_set_w_noise = ode_datas_wnoise[1]
###############
### time series #######
###############
fig = subfigs[1,2]
axs_ts = fig.subplots(3, 1, 
                sharex=true,           
                )
for g in 1:length(ranges)
    _tsteps = tsteps[ranges[g]]
    for i in 1:dim_prob
            ax = axs_ts[i]
                if g % 2 == 0 
                    ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:blue", alpha = 0.2) 
                else
                    ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:red", alpha = 0.2) 
                end
                ax.scatter(_tsteps, 
                        data_set_w_noise[i,ranges[g]], 
                        color = color_palette[i], 
                        # ls = lss[2], 
                        label =  (i == 1) && (g == 1) ? "Data" : nothing,
                        s = 10.)
                ax.set_ylim(-0.4,1.4)
                ax.set_yticklabels("")
                ax.set_xlim(tsteps[1]-2, tsteps[ranges[end][end]]+2)
    end
end
# fig.supxlabel("Time (days)")
fig.supylabel("Observation data")
axs_ts[end].set_xticklabels("")
[ax.set_facecolor("None") for ax in axs_ts]
display(masterfig)

fig = subfigs[2,2]
axs_sts = fig.subplots(3, 4, 
                # sharex=true,   
                gridspec_kw = Dict("wspace" => 0.6, "hspace"=>0.1)
                )

##############################
### splitted time series #####
##############################
for g in 1:length(ranges)
    _tsteps = tsteps[ranges[g]]
    axs_sts[1,g].annotate(L"L_\mathcal{M}^{(1,%$g)}(\theta)", (_tsteps[1], 1.8), annotation_clip=false, color= g % 2 == 1 ? "tab:red" : "tab:blue")
    for i in 1:dim_prob

            ax = axs_sts[i,g]
                if g % 2 == 0 
                    ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:blue", alpha = 0.2) 
                else
                    ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:red", alpha = 0.2) 
                end
                ax.scatter(_tsteps, 
                        data_set_w_noise[i,ranges[g]], 
                        color = color_palette[i], 
                        # ls = lss[2], 
                        label =  (i == 1) && (g == 1) ? "Data" : nothing,
                        s = 10.)
                ax.set_ylim(-0.4,1.4)
                ax.set_yticklabels("")
                ax.set_xlim(_tsteps[1]-2, _tsteps[end]+2)
                ax.set_xticklabels("")
    end
end
fig.supylabel("Mini-batches")
fig.supxlabel("Time", y = -0.04)
[ax.set_facecolor("None") for ax in axs_sts]

display(masterfig)

# con = matplotlib.patches.ConnectionPatch(xyA=(505,0), xyB=(505,1.4), coordsA="data", coordsB="data", 
#                       axesA=axs_ts[3], axesB=axs_sts[1], lw=1,  arrowstyle="->")
# axs_sts[1].add_artist(con)
display(masterfig)


_let = [L"\textbf{A}",L"\textbf{B}",L"\textbf{C}",L"\textbf{D}"]
for (i,ax) in enumerate([ax_loss, ax_its_loss, axs_ts[1]])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
    )
end
# masterfig.tight_layout()
# fig.savefig("figs/$name_scenario.pdf", dpi = 1000)


display(masterfig)


display(masterfig)

masterfig.savefig("figure1.png", dpi = 300,bbox_inches="tight")

