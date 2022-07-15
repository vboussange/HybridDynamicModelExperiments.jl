#=
Plotting two time series to illustrate the functioning of algorithm

Problem : difficult to put nice curly braces to emphasize on the chunks
=#
cd(@__DIR__)
using PyCall, PyPlot, EcologyInformedML, LaTeXStrings
py"""
import os
os.environ["PATH"]='/usr/local/opt/ruby/bin:/usr/local/lib/ruby/gems/3.0.0/bin:/Users/victorboussange/.gem/ruby/2.6.0/bin:/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/Applications/Visual Studio Code.app/Contents/Resources/app/bin:/usr/local/anaconda3/bin:/Applications/Julia-1.0.app/Contents/Resources/julia/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin'
"""
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")


rcParams["text.usetex"] = true
rcParams["text.latex.preamble"] = 
                                "\\usepackage{amsmath}
                                \\usepackage{DejaVuSans}"


using JLD2, UnPack
using EcologyInformedML:fontdict,lss
@load "ode_datas_wnoise.jld2"
@unpack losses, θs, ranges, p_labs = res

# plotting
h = 0.1
space = 0.01
fig,axs = plt.subplots(
            7,1,
            # constrained_layout=true,
            sharex = true,
            figsize = (5,7),
            gridspec_kw = Dict("height_ratios" => [h,h,h,space,h,h,h])
            )

#######################
#### time series ######
#######################
color_palette = ["tab:red", "tab:blue", "tab:green"]
dim_prob = size(pred[1],1)
for (k,data_set_w_noise) in enumerate(ode_datas_wnoise)
    idx_axs = (k-1)*4
    for g in 1:length(ranges)
        _pred = pred[g]
        _tsteps = tsteps[ranges[g]]
        for i in 1:dim_prob
                ax = axs[idx_axs+i]
                    if g % 2 == 0 
                        ax.fill_between(_tsteps, 0, 1.2, facecolor = "tab:grey", alpha = 0.2) 
                    else
                        ax.fill_between(_tsteps, 0, 1.2, facecolor = "black", alpha = 0.2) 
                    end
                    if k == 1 && i == 1
                        if g == length(ranges)
                            ax.annotate("\$\\overbrace{\\phantom{abcde}}^{y^{($g,$k)}}\$", (_tsteps[1], 0.9), fontsize = 15, annotation_clip=false)
                        elseif g % 2 == 0 
                            ax.annotate("\$\\overbrace{\\phantom{abcdef}}^{y^{($g,$k)}}\$", (_tsteps[1], 1.06), fontsize = 15, annotation_clip=false)
                        else
                            ax.annotate("\$\\overbrace{\\phantom{abcdef}}^{y^{($g,$k)}}\$", (_tsteps[1], 0.9), fontsize = 15, annotation_clip=false)
                        end               
                    elseif k == 2 && i == 3
                        if g == length(ranges)
                            ax.annotate("\$\\underbrace{\\phantom{abcde}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.16), fontsize = 15, annotation_clip=false)
                        elseif g % 2 == 0 
                            ax.annotate("\$\\underbrace{\\phantom{abcdef}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.06), fontsize = 15, annotation_clip=false)
                        else
                            ax.annotate("\$\\underbrace{\\phantom{abcdef}}_{y^{($g,$k)}}\$", (_tsteps[1], -0.16), fontsize = 15, annotation_clip=false)
                        end
                    end
                    ax.scatter(_tsteps, 
                            data_set_w_noise[i,ranges[g]], 
                            color = color_palette[i], 
                            # ls = lss[2], 
                            label =  (i == 1) && (g == 1) ? "Data" : nothing,
                            s = 20.)
                    ax.set_ylim(0.,1.2)
                    ax.set_yticklabels("")
                    ax.set_xlim(tsteps[1], tsteps[end])
        end
    end
end

# fig.axvline
axs[2].annotate("Time series 1",( -0.15,0.) , rotation = 90, annotation_clip=false, xycoords = "axes fraction")
axs[6].annotate("Time series 2",( -0.15,0.) , rotation = 90, annotation_clip=false, xycoords = "axes fraction")

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
display(fig)

fig.savefig("conceptual.png", dpi = 700)


# plot_convergence(p_init[dim_prob * trajectories + 1 : end], [p_init[(j-1)*dim_prob+1:j*dim_prob] for j in 1:trajectories], losses, θs, pred)
