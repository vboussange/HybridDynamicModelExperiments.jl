#=
Plotting results of model selection

AIC is calculated with all datapoints used in minibatch, with duplicates
=#
using FileIO, JLD2
using Statistics, LinearAlgebra, Distributions
using PyPlot, Printf
using DataFrames
using MiniBatchInference
using Glob
plotting = false

function get_results_AIC_omnivory_2sp(df_results)
    [df_results[!,stats] = fill(NaN,size(df_results,1)) for stats in ["loglikelihood", "AIC_likelihood", "AIC_RSS", "AICc_likelihood", "ΔAIC_likelihood", "ΔAICc_likelihood", "ΔRSS","W"]]
    println(count(df_results.training_success), " / ", size(df_results,1), " simulations were successful.\n Others deleted.")
    df_results = df_results[df_results.training_success .|> Bool,:] # deleting unsuccessfull columns
    for r in eachrow(df_results)
        k = length(r.parameters)
        m =  length(r.data_set) # nb of observations, which is number of state variables * number of time steps
        Σ = r.Σ[2:3,2:3] #only considering predator
        data_set_simu = cat(r.data_set_simu..., dims=2)
        data_set = cat([r.data_set[:,rng] for rng in r.res.ranges]..., dims=2)

        loglikelihood = sum(logpdf(MvNormal(data_set[2:3,i], Σ), data_set_simu[2:3,i]) for i in 1:size(data_set_simu,2) ) 
        AIC_likelihood = - 2 * loglikelihood + 2 * k # https://en.wikipedia.org/wiki/Akaike_information_criterion
        AICc_likelihood = AICc(AIC_likelihood, k, m)
        r[["loglikelihood", "AIC_likelihood", "AICc_likelihood"]] .= (loglikelihood, AIC_likelihood, AICc_likelihood)
    end

    dfg = groupby(df_results, ["noise"])
    for _df in dfg
        _dfg = groupby(_df,"ω")
        for __df in _dfg
            __df.ΔRSS .= __df.RSS .- minimum(__df.RSS)
            __df.ΔAIC_likelihood .= __df.AIC_likelihood .- minimum(__df.AIC_likelihood)
            __df.ΔAICc_likelihood .= __df.AICc_likelihood .- minimum(__df.AICc_likelihood)
        end
    end
    ## plotting AIC_likelihood vs ω, for different level of noises
    # println("*************** Standard model ***************")
    df_standard_model_array = []
    df_omnivory_model_array = []
    for idx in 1:length(dfg)
        df = dfg[idx]; sort!(df,"ω")
        println(df[1,"noise"])
        # selecting one of the model to plot

        # only taking best result of both simulations
        # TODO: to be checked and introduced in other script `analysis-omnivory.jl`
        df_standard_model = empty(df); df_omnivory_model = empty(df)
        scenarios = ["Standard model", "Omnivory model"]
        for (i,_df) in enumerate([df_standard_model, df_omnivory_model])
            __df = df[df.scenario .== scenarios[i],:]
            __dfg = groupby(__df, "ω")
            for ___df in __dfg
                myarg = argmin(___df.AIC_likelihood)
                println(myarg)
                push!(_df, ___df[myarg,:] )
            end
        end

        sort!(df_standard_model,"ω"); sort!(df_omnivory_model,"ω")
        @assert all(df_standard_model.ω .== df_omnivory_model.ω)
        df_standard_model.W .= exp.( .- df_standard_model.ΔAIC_likelihood / 2 ) ./ (exp.( .- df_standard_model.ΔAIC_likelihood / 2 ) .+ exp.(.- df_omnivory_model.ΔAIC_likelihood / 2))
        df_omnivory_model.W .= exp.( .- df_omnivory_model.ΔAIC_likelihood / 2 ) ./ (exp.( .- df_standard_model.ΔAIC_likelihood / 2 ) .+ exp.(.- df_omnivory_model.ΔAIC_likelihood / 2))

        println("We found $(count(df_standard_model.ΔAIC_likelihood .> 2.)) / $(size(df_standard_model,1)) points with ΔAIC_likelihood > 2")
        println("We found $(count(df_standard_model.ΔRSS .> 0.)) / $(size(df_standard_model,1)) points with positive ΔRSS")
        println(df[:,["scenario","ω","AIC_likelihood", "ΔAIC_likelihood", "RSS", "ΔRSS"]])
        push!(df_standard_model_array, df_standard_model)
        push!(df_omnivory_model_array, df_omnivory_model)
    end
    return df_standard_model_array, df_omnivory_model_array
end

##############################
# Simple plot of \Delta AIC
##############################
# fig, ax = subplots(1)
# ax.scatter(df_standard_model.ω, df_standard_model.AIC_likelihood, label = "r = $(dfg[idx].noise[1])")
# ax.set_title(dfg[idx].scenario[1])
# ax.legend()
# # ax.set_yscale("log")
# display(fig)

##############################
# plotting of \Delta AIC
##############################
if plotting 
    cd(@__DIR__)

    dir_res = "2022-07-04/"
    name_scenario = "omnivory-hypothesis_testing_v2_simple_minibatch_step_4_datasize_40_1sp"
    @load "$(dir_res)/$(name_scenario).jld2" df_results
    df_standard_model_array, df_omnivory_model_array = get_results_AIC_omnivory_1sp(df_results)
    fig, axs = subplots(2, 3, sharex="col", sharey="row", gridspec_kw = Dict("height_ratios" => [2,1]))
    # filling areas for weak support
    ωspan = [-0.025, 0.525]
    ylim = 3000
    for ax in axs
        ax.set_xlim(ωspan) # ω span
        ax.fill_between(ωspan, 0, 2, facecolor = "tab:grey", alpha = 0.1)
        ax.fill_between(ωspan, 4, 8, facecolor = "tab:grey", alpha = 0.3)
        ax.fill_between(ωspan, 8, ylim, facecolor = "tab:grey", alpha = 0.8)
    end

    axs[2].annotate("no support",[0.1, 1.])
    axs[2].annotate("weak support",[0.1, 6.])
    axs[1].annotate("strong support",[0.1, ylim*0.7])
    gcf()

    # Plotting
    colors = ["tab:blue", "tab:orange"]
    hyp_lab = ["Omnivory hypothesis", "No omnivory hypothesis", ]
    for (m,df_model_array) in enumerate([df_standard_model_array,df_omnivory_model_array])
        for (i,_df) in enumerate(df_model_array[1:2:end])
            axs[1,i].set_title(L"r = %$(_df.noise[1])")
            axs[1,i].scatter(_df.ω, _df[:,:ΔAICc_likelihood], c  = colors[m], label = i == 1 ? hyp_lab[m] : nothing )
            axs[1,i].set_yscale("log")

            axs[2,i].scatter(_df.ω, _df[:,:ΔAICc_likelihood], c  = colors[m])
        end
    end
    axs[2,2].set_xlabel("Strength of omnivory ("*L"\omega"*")")
    fig.legend(loc = "upper center", bbox_to_anchor=(0.3, 1.1),)
    fig.tight_layout()
    gcf()

    # fig.supylabel(L"\Delta"*"AIC\n",fontsize=15)
    axs[2].set_ylim(-0.5,8)
    axs[1].set_ylim(8,ylim)
    display(fig)
    fig.savefig(dir_res*"AIC_likelihood_comparision_$name_scenario.png", dpi=300)
end