#=
Figure 1 of main manuscript.

=#
cd(@__DIR__)
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using Printf
using DataFrames
using PiecewiseInference
using Glob
using UnPack
using JLD2
using LaTeXStrings
include("../format.jl")
# LATEX needed

@py begin
    import os
    os.environ["PATH"]= "/opt/homebrew/Caskroom/mambaforge/base/bin:/opt/homebrew/Caskroom/mambaforge/base/condabin:/Users/victorboussange/.juliaup/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin" # path to repo of binary with latex
end
rcParams["text.usetex"] = true
rcParams["text.latex.preamble"] = 
                                "\\usepackage{amsmath}
                                \\usepackage{DejaVuSans}
                                \\usepackage{sansmath, amsfonts}
                                \\sansmath
                                \\usepackage{MnSymbol}
                                \\usepackage{nicefrac}
                                " # MnSymbol used for tiny braces, see https://latex.org/forum/viewtopic.php?t=21137
color_palette = ["tab:red", "tab:blue", "tab:green",]
linestyles = ["solid", "dashed", "dotted"]

# DATA FOR LOSSES AND TIME SERIES
results_convergence = load("../../scripts/illustrate_convergence/illustrate_convergence.jld2")

data_set_w_noise, tsteps, ranges = load("../../scripts/illustrate_convergence/data_set_w_noise.jld2", "data_set_w_noise", "tsteps", "ranges")

# DATA FOR PROFILE LOSS
profile_loss = load("../../scripts/profile_likelihood/profile_likelihood.jld2")

dim_prob = 3 #ODE dimension

masterfig = plt.figure(figsize = (5.2,5))
masterfig.set_facecolor("None")
subfigs = masterfig.subfigures(2, 2, wspace=0.1, width_ratios=[1.,1.5])
# [fig.set_facecolor("None") for fig in pyconvert(Vector,subfigs)]

##########################
### plotting profile loss ###
##########################
ax_loss = subfigs[0,0].subplots()
# ax.set_title("Time horizon T = $(tsteps[end] - tsteps[1])")
ax_loss.plot(profile_loss["params"], profile_loss["p_likelihood_piecewise"], label = L"L_{\mathcal{M}}^\star", linestyle = linestyles[2],c="tab:orange")
ax_loss.plot(profile_loss["params"], profile_loss["p_likelihood"], label = L"L_{\mathcal{M}}", linestyle = linestyles[1],c="tab:blue")
ax_loss.axvline(profile_loss["p_true"],  label = L"\Tilde r_3", linestyle = linestyles[3], color = "red")
# ax_loss.set_ylabel(L"L(p_2)")
ax_loss.legend(loc="lower right")
ax_loss.set_xlabel(L"r_3",y=0.2,zorder=10)
ax_loss.set_yscale("log")
display(masterfig)


################################
# plotting loss vs iterations  #
################################
p_err_dict = Dict()
for k in ["batch_inference", "naive_inference_small_p", "naive_inference"]
    p_true = results_convergence["p_true"]
    perr = [median(abs.((p_true .- p) ./ p_true)) for p in results_convergence[k]]
    p_err_dict[k] = perr
end
ax_its_loss = subfigs[1,0].subplots()
ax_its_loss.plot(1:length(p_err_dict["batch_inference"])-2, p_err_dict["batch_inference"][1:end-2], label = L"L_{\mathcal{M}}^\star", linestyle = linestyles[2],c="tab:orange")
ax_its_loss.plot(1:length(p_err_dict["naive_inference"])-2, p_err_dict["naive_inference"][1:end-2], label = L"L_{\mathcal{M}}", linestyle = linestyles[1],c="tab:blue")
ax_its_loss.plot(1:length(p_err_dict["naive_inference_small_p"])-2, p_err_dict["naive_inference_small_p"][1:end-2], label = L"L_{\mathcal{M}}, \theta_0 = \Tilde{\theta} + \delta\theta", linestyle = linestyles[3],c="tab:green")
ax_its_loss.legend(loc="upper right")
# ax_its_loss.set_yscale("log")
ax_its_loss.set_ylabel(L"|\nicefrac{(\hat p -\Tilde{p})}{\Tilde{p}}|")

ax_its_loss.set_xlabel("Epochs")
# ax_its_loss.set_yscale("log")
[ax.set_facecolor("None") for ax in [ax_its_loss,ax_loss]]
display(masterfig)


###############
### time series #######
###############
fig = subfigs[0,1]
axs_ts = fig.subplots(3, 1, 
                sharex=true,           
                )
for g in 1:length(ranges)
    _tsteps = tsteps[ranges[g]]
    for i in 1:dim_prob
            ax = axs_ts[i-1]
            if i == 1 && g==1
                ax.annotate("\$\\overbrace{\\phantom{abcde}}^{S}\$", (_tsteps[1], 1.), fontsize = 15, annotation_clip=false)
                ax.annotate("\$\\overbrace{\\phantom{abcdeabcdeabcdeabc}}^{K+1}\$", (_tsteps[1], 0.8), fontsize = 15, annotation_clip=false)
            end
            if i == 3 && g==1
                ax.annotate("\$\\underbrace{\\phantom{\\footnotesize{abcd}}}_{R}\$", (_tsteps[1], -0.5), fontsize = 15, annotation_clip=false)
            end
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
            # ax.set_yscale("log")
            ax.set_yticklabels("")
            ax.set_xlim(tsteps[1]-2, tsteps[ranges[end][end]]+2)
    end
end
# fig.supxlabel("Time (days)")
fig.supylabel("Species abundance")
axs_ts[-1].set_xticklabels("")
[ax.set_facecolor("None") for ax in axs_ts]
display(masterfig)

fig = subfigs[1,1]
axs_sts = fig.subplots(3, 4, 
                # sharex=true,   
                gridspec_kw = Dict("wspace" => 0.6, "hspace"=>0.1)
                )

##############################
### splitted time series #####
##############################
for g in 1:length(ranges)
    _tsteps = tsteps[ranges[g]]
    axs_sts[0,g-1].annotate(L"L_\mathcal{M}^{(%$g)}(\theta)", (_tsteps[1], 1.8), annotation_clip=false, color= g % 2 == 1 ? "tab:red" : "tab:blue")
    for i in 1:dim_prob

            ax = axs_sts[i-1,g-1]
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
                # ax.set_yscale("log")
                ax.set_yticklabels("")
                ax.set_xlim(_tsteps[1]-2, _tsteps[end]+2)
                ax.set_xticklabels("")
    end
end
fig.supylabel("Species abundance")
fig.supxlabel("Time", y = -0.04)

display(masterfig)


_let = [L"\textbf{A}",L"\textbf{B}",L"\textbf{C}",L"\textbf{D}"]
for (i,ax) in enumerate([ax_loss, ax_its_loss])
    _x = -0.1
    ax.text(_x, 1.05, _let[i],
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        transform=ax.transAxes ,
    )
end
axs_ts[1].text(-0.1, 2.4, L"\textbf{C}",
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
            transform=axs_ts[1].transAxes ,
            )
# fig.savefig("figs/$name_scenario.pdf", dpi = 1000)


# masterfig.tight_layout()
display(masterfig)
masterfig.savefig("figure1.pdf", dpi = 300,bbox_inches="tight")

