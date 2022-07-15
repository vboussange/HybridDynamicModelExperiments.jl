#=

Here we vary some parameters of the model and we perform inference, for one time series only.
The goal is 
    - to show the performance of the inference scheme
    - to relate the performance of the scheme with the information content

* Version
- 1sp : only one species fitted.

TODO: u0s of untracked species are initialised with dataset, which is a bit cheating (although it probably does not play an important role)
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
using Random; Random.seed!(101)

include("../model/composable_ecosystem_model.jl")

function loss_1sp(data, pred, ic_term)
    l =  mean((data[:,:] - pred[2:3,:]).^2)
    l +=  mean((data[:,1] - pred[2:3,1]).^2) * ic_term # putting more weights on initial conditions
    return l
end

if length(ARGS) > 0
    @unpack noise, datasize, step = parse_commandline()
else
    steps = [4]#[2, 4, 8]
    datasizes = [40,60,80] #10:10:40 #[50, 100]
    noises = 0:0.2:1.0
    xp_s = range(0.071, 0.225, length = 50)
end
_today = today()

#################
### plotting ####
#################

u0_true = [0.5,0.8,0.5]
name_scenario = "3-species-model_McKann_simple_minibatch_2sp"

verbose = true
info_per_its = 200
plot_loss = false
loop = true

alg = Tsit5()
threshold = -1e99 # threshold for stopping optimisation
sensealg = ForwardDiffSensitivity()
abstol = 1e-6
reltol=1e-6

###################
# learning params #
###################
optimizers = [ADAM(1e-1), ADAM(1e-2), ADAM(1e-3), BFGS(initial_stepnorm=0.001)]
maxiters = [2000, 2000, 2000, 200]
group_size_init = 10 + 1 # number of points in each segment for multiple shooting
continuity_term = 0. #200. is the standard value. The more the noise, the less the continuity term
ic_term = 1 / group_size_init

p_labs = [L"x_c", L"x_p", L"y_c", L"y_p", L"R_0", L"C_0"]

#############################
#### starting looping #######
#############################
if plot_loss 
    using PyPlot # required for launching plotting 
end

function simu(pars)
    @unpack noise, datasize, step, x_p = pars
    println("***************\nSimulation started for x_p = $x_p,\n noise level r= $noise,\n datasize = $datasize,\n step = $step\n***************\n")

    em_true = EcosystemModelMcKann(x_c = 0.4, 
                                    x_p = x_p, 
                                    y_c = 2.01, 
                                    y_p = 5.00, 
                                    R_0 = 0.16129, 
                                    C_0 = 0.5)

    # initialising parameters
    em_init = EcosystemModelMcKann(x_c = em_true.p[1] * (2. *rand()), 
                                    x_p = 0.4 * (2. *rand()), 
                                    y_c = em_true.p[3] * (2. *rand()),  
                                    y_p = em_true.p[4] * (2. *rand()), 
                                    R_0 = em_true.p[5] * (2. *rand()),  
                                    C_0 = em_true.p[6] * (2. *rand()))

    ########################
    #### generating data ###
    ########################
    tsteps = range(500., step=step, length = datasize) # we discard transient dynamics, only considering dynamics from 500 to 700
    tspan = (0.,tsteps[end])

    prob = ODEProblem(em_true, u0_true, tspan, em_true.p) # needed for generating data 
    sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
    # using Plots
    # Plots.plot(sol_data)
    ## Ideal data
    data_set = Array(sol_data)
    
    prob_long = ODEProblem(em_true, u0_true, (0., 1000.), em_true.p) # needed for generating a long time series to obtain variance of variables
    long_data = solve(prob_long, alg, saveat = 500.:1000., abstol=abstol, reltol=reltol, sensealg = sensealg)

    # Data with noise
    std_noise = reshape(std(long_data, dims=2), :) .* noise
    data_set_w_noise = data_set + randn(size(data_set)...)  .* std_noise
    Σ = diagm(std_noise.^2)
    
    # calculating information content
    if noise !== 0.
        fim = FIM_yazdani(em_true, data_set[:,1], (tsteps[1], tsteps[end]), tsteps, em_true.p, Σ) # TODO: this is probably false, one needs to consider the only availability of predators
    else
        fim = [] # FIM undefined
    end

    # initialising parameters, but those get overridden by `minibatch_MLE`
    p_init = em_init.p
    prob_simul = ODEProblem(em_true, data_set_w_noise[:,1], (tsteps[1], tsteps[end]), p_init)

    # here we manually initialize initial conditions and ranges, as the training data does not correspond to the observables
    ranges = DiffEqFlux.group_ranges(datasize, group_size_init)
    u0s_init = data_set_w_noise[:,first.(ranges),:]
    # u0s_init[1:2,:] .= 0.5 # initialising to 0.5 ICs for which data is not available
    u0s_init = reshape(u0s_init,:)
    data_set_w_noise = data_set_w_noise[2:3,:] #only predator and consumers are considered  
    
    stats = @timed EcologyInformedML._minibatch_MLE(ranges = ranges,
                                optimizers = optimizers,
                                p_init = p_init,
                                data_set = data_set_w_noise, 
                                prob = prob_simul, 
                                tsteps = tsteps, 
                                alg = alg, 
                                sensealg = sensealg, 
                                maxiters = maxiters,
                                p_true = em_true.p,
                                p_labs = p_labs,
                                continuity_term = continuity_term,
                                ic_term = ic_term,
                                verbose = verbose,
                                info_per_its = info_per_its,
                                plotting = plot_loss,
                                threshold = threshold,
                                loss_fn = loss_1sp,
                                u0s_init = u0s_init
                                )
    res = stats.value; simtime = stats.time
    [println(L"\hat{%$(res.p_labs[i])} = ", abs(res.p_trained[i]), L", %$(res.p_labs[i]) = ", res.p_true[i]) for i in 1:length(res.p_true)]
    return (res, x_p, abs.(res.p_trained), res.p_true, data_set, data_set_w_noise, res.pred, res.minloss, noise, step, datasize, Σ, fim, length(res.ranges), simtime)
end

# initialising df and pars
pars_arr = Dict{String,Any}[]

df_results = DataFrame( "res" => ResultMLE[],
                        "x_p" => [],
                        "p_trained" => [], 
                        "p_true" => [], 
                        "data_set_clean" => [],
                        "data_set_training" => [],
                        "data_set_simu" => [],
                        "RSS" => [], 
                        "noise" => [], 
                        "step" => [],
                        "datasize" => [],
                        "Σ" => [],
                        "FIM" => [],
                        "ngroups" => [],
                        "simtime" => [],
                        "training_success" => [],
                        )


for step in steps
    for datasize in datasizes
        for noise in noises
            for x_p in xp_s
                pars = Dict{String,Any}()
                @pack! pars = noise, step, datasize, x_p
                push!(pars_arr, pars)
                push!(df_results, (ResultMLE(), NaN, [], [], [], [],[], [], 0., [], [], [], [], [], [], false))
            end
        end
    end
end

# Trying simul function
# verbose = true
if !loop # for debugging
    df_results[1,:] = (simu(pars_arr[1])...,true)
    df_results[2,:] = (simu(pars_arr[2])...,true)
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
end


dict_simul = Dict{String, Any}()
@pack! dict_simul = df_results
save(joinpath("results",string(_today), "$name_scenario.jld2"), dict_simul)
println("Results saved")