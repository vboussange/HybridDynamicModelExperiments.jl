#=
Figure 1 in main manuscript.
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
# DATA FOR LOSSES AND TIME SERIES
res_batch = load("results/2022-06-20/batch_inference.jld2", "res") 
res_naive = load("results/2022-06-20/naive_inference.jld2", "res") 
res_naive_neigh = load("results/2022-06-20/naive_inference_neighbourhood.jld2", "res") 

ode_datas_wnoise, tsteps, ranges = load("ode_datas_wnoise.jld2", "ode_datas_wnoise", "tsteps", "ranges")

# DATA FOR PROFILE LOSS
@load "../../continuity_analysis/plikelihood_datasize=200.jld2"

dim_prob = 3 #ODE dimension

masterfig = plt.figure(figsize = (8,6),constrained_layout=true)
subfigs = masterfig.subfigures(1, 2, wspace=0.1, width_ratios=[1.,1.5])

##########################
### plotting profile loss ###
##########################
ax_loss, ax_its_loss = subfigs[1].subplots(2,1)
# ax.set_title("Time horizon T = $(tsteps[end] - tsteps[1])")
ax_loss.plot(x_ps, p_likelihood, label = L"L_{\mathcal{M}}(\theta)", linestyle = linestyles[1])
ax_loss.plot(x_ps, p_likelihood_EIML, label = L"L_{\mathcal{M}}^\star(\theta)", linestyle = linestyles[2])
ax_loss.axvline(p_true,  label = L"\Tilde p_2", linestyle = linestyles[3], color = "red")
ax_loss.set_xlabel(L"p_2")
# ax_loss.set_ylabel(L"L(p_2)")
ax_loss.legend(loc="best", bbox_to_anchor=(0.5, 0.45, 0.4, 0.5))
display(masterfig)


################################
# plotting loss vs iterations  #
################################
ax_its_loss.plot(1:length(res_batch.θs)-1, res_batch.θs[1:end-1], label = L"L_{\mathcal{M}}^\star(\theta)", linestyle = linestyles[1])
ax_its_loss.plot(1:length(res_naive.θs)-1, res_naive.θs[1:end-1], label = L"L_{\mathcal{M}}(\theta)", linestyle = linestyles[2])
ax_its_loss.plot(1:length(res_naive_neigh.θs)-1, res_naive_neigh.θs[1:end-1], label = L"L_{\mathcal{M}}(\Tilde{\theta} + \delta\theta)", linestyle = linestyles[3])
ax_its_loss.legend(loc="lower right")
ax_its_loss.set_yscale("log")
ax_its_loss.set_xlabel("Iterations")
ax_its_loss.set_ylabel(L"\|\theta_m - \Tilde\theta \|", fontsize=14,)

###############
### time series #######
###############
h = 0.1
space = 0.01
fig = subfigs[2]
axs = fig.subplots(7, 1, 
                sharex=true,           
                gridspec_kw = Dict("height_ratios" => [h,h,h,space,h,h,h])
                )


#######################
#### time series ######
#######################
color_palette = ["tab:red", "tab:blue", "tab:green"]
for (k,data_set_w_noise) in enumerate(ode_datas_wnoise)
    idx_axs = (k-1)*4
    for g in 1:length(ranges)
        _tsteps = tsteps[ranges[g]]
        for i in 1:dim_prob
                ax = axs[idx_axs+i]
                    if g % 2 == 0 
                        ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:blue", alpha = 0.2) 
                    else
                        ax.fill_between([_tsteps[1] - 2;_tsteps[end] + 2], -0.4, 1.4, facecolor = "tab:red", alpha = 0.2) 
                    end
                    # if k == 1 && i == 1
                    #     if g == length(ranges)
                    #         ax.annotate("\$\\overbrace{\\phantom{abcde}}^{y^{($g,$k)}}\$", (_tsteps[1], 0.9), fontsize = 15, annotation_clip=false)
                    #     elseif g % 2 == 0 
                    #         ax.annotate("\$\\overbrace{\\phantom{abcdef}}^{y^{($g,$k)}}\$", (_tsteps[1], 1.06), fontsize = 15, annotation_clip=false)
                    #     else
                    #         ax.annotate("\$\\overbrace{\\phantom{abcdef}}^{y^{($g,$k)}}\$", (_tsteps[1], 0.9), fontsize = 15, annotation_clip=false)
                    #     end               
                    # elseif k == 2 && i == 3
                    #     if g == length(ranges)
                    #         ax.annotate("\$\\underbrace{\\phantom{abcde}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.16), fontsize = 15, annotation_clip=false)
                    #     elseif g % 2 == 0 
                    #         ax.annotate("\$\\underbrace{\\phantom{abcdef}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.06), fontsize = 15, annotation_clip=false)
                    #     else
                    #         ax.annotate("\$\\underbrace{\\phantom{abcdef}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.16), fontsize = 15, annotation_clip=false)
                    #     end
                    # end
                    ax.scatter(_tsteps, 
                            data_set_w_noise[i,ranges[g]], 
                            color = color_palette[i], 
                            # ls = lss[2], 
                            label =  (i == 1) && (g == 1) ? "Data" : nothing,
                            s = 20.)
                    ax.set_ylim(-0.4,1.4)
                    ax.set_yticklabels("")
                    ax.set_xlim(tsteps[1]-2, tsteps[end]+2)
        end
    end
end

# fig.axvline
axs[2].annotate("Time series 1",( -0.1,0.) , rotation = 90, annotation_clip=false, xycoords = "axes fraction")
axs[6].annotate("Time series 2",( -0.1,0.) , rotation = 90, annotation_clip=false, xycoords = "axes fraction")

# labs_ts = ["Resource R", "Consumer C", "Predator P"]
# for (i, lab) in enumerate(labs_ts)
#     for ax in [axs[i], axs[i+4]]
#         ax.set_ylabel(lab, fontsize=12)
#     end
# end
# axs[1].legend(fontsize=12, bbox_to_anchor=[0.3, 1.1])
# axs[1].set_clip_on(true)
axs[3].set_xlabel("Time (days)")
axs[end].set_xticklabels("")

axs[4].axis("off")
# fig.tight_layout()
display(masterfig)


_let = [L"\textbf{A}",L"\textbf{B}",L"\textbf{C}",L"\textbf{D}"]
for (i,ax) in enumerate([ax_loss, ax_its_loss, axs[1]])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
    )
end
# masterfig.tight_layout()
# fig.savefig("figs/$name_scenario.pdf", dpi = 1000)
display(masterfig)

masterfig.savefig("ML_algo.png", dpi = 300,bbox_inches="tight")

