#=
    perform model selection from a bank of candidate models

* Version
EcosystemModelOmnivory

- using simple minibatching
- only varying noise and omega, with fixed length size 

* Results
- name_scenario = "omnivory-hypothesis_testing_2" :
    - used the modified version of EcosystemModelMcKann
    - lowered down the group_size_init
- 2022-04-15
    - decreased step=2 and group_init=10
- 2022-04-16
    - increased step=4 and group_init=5
    - optimizers = [ADAM, BFGS],
    - continuity term = 0. # smoothens out much more nicely the loss curve
- 2022-07-11
    - exp(0.5) for params init
- 2022-07-11
    - 2 unif for params init
=#
cd(@__DIR__)

using FileIO, JLD2
using Statistics
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using LaTeXStrings
using UnPack, ProgressMeter
using EcologyInformedML
using Revise
import Random; Random.seed!(10)

include("../model/composable_ecosystem_model.jl")
include("../inference_exploration/parse_utils.jl") # should be moved to ecosystem model at some point
if length(ARGS) > 0
    @unpack noise, datasize, step = parse_commandline()
else
    steps = [4]#[2, 4, 8]
    datasizes = [60] #[50, 100]
    noises =  0.1:0.1:0.5
    ωs = 0.:0.025:0.5 #0:0.02:0.24
end
_today = today()

name_scenario = "omnivory-hypothesis_testing_v2_simple_minibatch_step_4_datasize_60_allsp"

u0_true = [0.8,0.8,2.]

verbose = true
info_per_its = 400
plotting = false
loop = true
nruns = 3

alg = Tsit5()
threshold = -1e99 # threshold for stopping optimisation
sensealg = ForwardDiffSensitivity()
abstol = 1e-6
reltol=1e-6

###################
# learning params #
###################
optimizers = [ADAM(1e-2), ADAM(1e-3), BFGS(initial_stepnorm=0.001)]
maxiters = [2000, 4000, 200]
group_size_init = 10 + 1 # number of points in each segment for multiple shooting
continuity_term = 0. #200. is the standard value. The more the noise, the less the continuity term
ic_term = 1 / group_size_init

if plotting 
    using PyPlot # required for launching plotting 
end

function simu(pars)
    @unpack em_true, em, scenario, noise, datasize, step, ω = pars

    ########################
    #### generating data ###
    ########################
    tsteps = range(500., step=step, length = datasize) # we discard transient dynamics, only considering dynamics from 500 to 700
    tspan = (0.,tsteps[end])

    prob = ODEProblem(em_true, u0_true, tspan, em_true.p) # needed for generating data 
    sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
    # using Plots
    # Plots.scatter(sol_data)
    # Ideal data
    data_set = Array(sol_data)

    prob_long = ODEProblem(em_true, u0_true, (0., 1000.), em_true.p) # needed for generating a long time series to obtain variance of variables
    long_data = solve(prob_long, alg, saveat = 500.:1000., abstol=abstol, reltol=reltol, sensealg = sensealg)

    # Data with noise
    std_noise = reshape(std(long_data, dims=2), :) .* noise
    data_set_w_noise = data_set + randn(size(data_set)...)  .* std_noise
    Σ = diagm(std_noise.^2)
    # Plots.scatter(data_set_w_noise')


    # calculating information content
    if noise !== 0.
        fim = FIM_yazdani(em_true, data_set[:,1], (tsteps[1], tsteps[end]), tsteps, em_true.p, Σ)
    else
        fim = [] # FIM undefined
    end

    
    # initialising parameters, but those get overridden by `minibatch_MLE`
    p_init = em.p
    prob_simul = ODEProblem(em, data_set_w_noise[:,1], (tsteps[1], tsteps[end]), p_init)

    println("$scenario")
    println("***************\nSimulation started for noise level r= $noise,\n datasize = $datasize,\n step = $step\n***************\n")

    
    stats = @timed minibatch_MLE(group_size = group_size_init,
                                optimizers = optimizers,
                                p_init = p_init,
                                data_set = data_set_w_noise, 
                                prob = prob_simul, 
                                tsteps = tsteps, 
                                alg = alg, 
                                sensealg = sensealg, 
                                maxiters = maxiters,
                                continuity_term = continuity_term,
                                ic_term = ic_term,
                                verbose = verbose,
                                plotting = plotting,
                                info_per_its = info_per_its,
                                )
    res = stats.value; simtime = stats.time
    return (res, ω, step, datasize, scenario, abs.(res.p_trained), data_set, data_set_w_noise, res.pred, res.minloss, noise, Σ, length(res.ranges), fim,simtime)
end

# initialising df and pars
pars_arr = Dict{String,Any}[]

df_results = DataFrame("res" => ResultMLE[],
                        "ω" => [],
                        "step" => [],
                        "datasize" =>[],
                        "scenario" => [],
                        "parameters" => [], 
                        "data_set" => [],
                        "data_set_training" => [],
                        "data_set_simu" => [],
                        "RSS" => [], 
                        "noise" => [], 
                        "Σ" => [],
                        "ngroups" => [],
                        "simtime" => [],
                        "FIM" => [],
                        "training_success" => [],
                        )

for step in steps
    for datasize in datasizes
        for noise in noises
            for ω in ωs
                for i in 1:nruns

                    # initial conditions are take from section 4 of mc kann 1994, "limit cycles"
                    em_true = EcosystemModelOmnivory(x_c = 0.4, 
                                                    x_p = 0.08, 
                                                    y_c = 2.009, 
                                                    y_pr = 2.00, 
                                                    y_pc = 5., 
                                                    R_0 = 0.16129, 
                                                    R_02 = 0.5, 
                                                    C_0 = 0.5, 
                                                    ω = ω)

                    ##################
                    # ecosystem bank #
                    ##################

                    # initial conditions are take from section 4 of mc kann 1994, "limit cycles"
                    em_omnivory = EcosystemModelOmnivory(x_c = em_true.p[1] * (2. *rand()), 
                                                        x_p = em_true.p[2] * (2. *rand()), 
                                                        y_c = em_true.p[3] * (2. *rand()),  
                                                        y_pc = em_true.p[4] * (2. *rand()),  
                                                        y_pr = em_true.p[5] * (2. *rand()),  
                                                        R_0 = em_true.p[6] * (2. *rand()),  
                                                        R_02 = em_true.p[7] * (2. *rand()),  
                                                        C_0 = em_true.p[8] * (2. *rand()),
                                                        ω = 0.25 * (2. *rand()))

                    em_std = EcosystemModelMcKann(x_c = em_true.p[1] * (2. *rand()), 
                                                    x_p = em_true.p[2] * (2. *rand()), 
                                                    y_c = em_true.p[3] * (2. *rand()),  
                                                    y_p = em_true.p[5] * (2. *rand()), 
                                                    R_0 = em_true.p[6] * (2. *rand()),  
                                                    C_0 = em_true.p[8] * (2. *rand()))

                    mybank = Dict(["Omnivory model", "Standard model",] .=> [em_omnivory, em_std])

                    for em_pair in mybank
                        pars = Dict("scenario" => em_pair[1], "em" => em_pair[2], "em_true"=>em_true)
                        @pack! pars = noise, step, datasize, ω
                        push!(pars_arr, pars)
                        push!(df_results, (ResultMLE(), [], [], [],"", [], [], [], [], 0., [], [],0,[],[], false))
                    end
                end
            end
        end
    end
end
# Trying simul function
# verbose = true
if !loop # for debugging
    df_results[1,:] = (simu(pars_arr[1])...,true)
    df_results[2,:] = (simu(pars_arr[2])...,true)
    println(df_results[1:2,["scenario","RSS","ngroups"]])
else
    progr = Progress(length(pars_arr), showspeed = true, barlen = 10)
    for k in 1:length(pars_arr)
        try
            df_results[k,:] = (simu(pars_arr[k])..., true);
        catch e
            println("problem with p = $(pars_arr[k])")
            println(e)
        end
        next!(progr)
    end
    println(df_results[:,["scenario","RSS","ngroups"]])
end


dict_simul = Dict{String, Any}()
@pack! dict_simul = df_results
save(joinpath("results",string(_today), "$name_scenario.jld2"), dict_simul)
println("Results saved")