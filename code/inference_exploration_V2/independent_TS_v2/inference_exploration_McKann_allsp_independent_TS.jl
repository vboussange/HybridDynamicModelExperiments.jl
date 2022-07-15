#=

Here we vary some parameters of the model and we perform inference, for one time series only.
The goal is 
    - to show the performance of the inference scheme
    - to relate the performance of the scheme with the information content

* Version
- all : we introduce a parser to take arguments as a dictionary
- McKann : we use McKann reasonable parameters
- independent_TS: exploring the combination of independent time series
* Results
- 2022-07-04: 30 datapoints
- 2022-07-05: 100 datapoints
- 2022-07-06: 100 datapoints, σ params = params
- 2022-07-12: 100 datapoints, exp distrib param init
- 2022-07-14: 50 datapoints, exp distrib param init
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
using Random; Random.seed!(1012)

include("../../model/composable_ecosystem_model.jl")

if length(ARGS) > 0
    @unpack noise, datasize, step = parse_commandline()
else
    steps = [4]#[2, 4, 8]
    nb_tss = 1:6
    datasize = 12 #[50, 100]
    noises = 0.0:0.2:1.0
    xp_s = range(0.071, 0.225, length = 50)
end
_today = today()

#################
### plotting ####
#################

u0s_true = [rand(3) for i in 1:maximum(nb_tss)]
name_scenario = "3-species-model_McKann_simple_minibatch_allsp_independent_TS"

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
group_size_init = 6 + 1 # number of points in each segment for multiple shooting
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
    @unpack noise, nb_ts, step, x_p = pars
    println("***************\nSimulation started for x_p = $x_p,\n noise level r= $noise,\n nb_ts = $nb_ts,\n step = $step\n***************\n")

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
    tsteps = [range(500., step=step, length = datasize) for i in 1:nb_ts] # we discard transient dynamics, only considering dynamics from 500 to 700

    data_sets = []
    for i in 1:nb_ts
        tspan = (0.,tsteps[i][end])
        prob = ODEProblem(em_true, u0s_true[i], tspan, em_true.p)
        sol_data = solve(prob, alg, saveat = tsteps[i], abstol=abstol, reltol=reltol, sensealg = sensealg)
        # using Plots
        # Plots.scatter(sol_data)
        # Ideal data
        data_set = Array(sol_data)
        push!(data_sets,data_set)
    end
    
    # generating a long time series to obtain variance of variables
    prob_long = ODEProblem(em_true, u0s_true[1], (0., 1000.), em_true.p)
    long_data = solve(prob_long, alg, saveat = 500.:1000., abstol=abstol, reltol=reltol, sensealg = sensealg)

    # Data with noise
    std_noise = reshape(std(long_data, dims=2), :) .* noise
    data_sets_w_noise = [data_set + randn(size(data_set)...)  .* std_noise for data_set in data_sets]
    Σ = diagm(std_noise.^2)
    
    # calculating information content
    if noise !== 0.
        fim = sum([FIM_yazdani(em_true, data_sets[i][:,1], (tsteps[i][1], tsteps[i][end]), tsteps[i], em_true.p, Σ) for i in 1:nb_ts])
    else
        fim = [] # FIM undefined
    end

    p_init = em_init.p
    prob_simul = ODEProblem(em_true, data_sets_w_noise[1][:,1], (tsteps[1], tsteps[end]), em_true.p) # needed for simulations, to only start at tsteps[1] and not 0.

    
    stats = @timed minibatch_ML_indep_TS(group_size = group_size_init,
                                optimizers = optimizers,
                                p_init = p_init,
                                data_set = data_sets_w_noise, 
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
                                threshold = threshold
                                )
    res = stats.value; simtime = stats.time
    [println(L"\hat{%$(res.p_labs[i])} = ", abs(res.p_trained[i]), L", %$(res.p_labs[i]) = ", res.p_true[i]) for i in 1:length(res.p_true)]
    return (res, x_p, abs.(res.p_trained), res.p_true, data_sets, data_sets_w_noise, res.pred, res.minloss, noise, step, datasize, nb_ts, Σ, fim, length(res.ranges), simtime)
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
                        "nb_ts" => [],
                        "Σ" => [],
                        "FIM" => [],
                        "ngroups" => [],
                        "simtime" => [],
                        "training_success" => [],
                        )


for step in steps
    for nb_ts in nb_tss
        for noise in noises
            for x_p in xp_s
                pars = Dict{String,Any}()
                @pack! pars = noise, step, nb_ts, x_p
                push!(pars_arr, pars)
                push!(df_results, (ResultMLE(), NaN, [], [], [], [],[], [], 0., [], [], [], [], [], [], [], false))
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