#=

Parameter inference, for one time series only, 
with the naive implementation
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
using MiniBatchInference
using Revise
include("../../parameter_inference/3-species-model/model/composable_ecosystem_model.jl")

datasize = 200
step = 4
noise = 0.2
x_p = 0.09
_today = today()

u0_true = [0.8,0.8,0.8]
name_scenario = "naive_inference"

verbose = true
info_per_its = 200
plot_loss = true

alg = Tsit5()
threshold = -1e99 # threshold for stopping optimisation
sensealg = ForwardDiffSensitivity()
abstol = 1e-6
reltol=1e-6

###################
# learning params #
###################
optimizers = [ADAM(5e-3)]
maxiters = [5000]
group_size_init = 200 # number of points in each segment for multiple shooting
continuity_term = 0. #200. is the standard value. The more the noise, the less the continuity term
ic_term = 1.

p_labs = [L"x_c", L"x_p", L"y_c", L"y_p", L"R_0", L"C_0"]

#############################
#### starting looping #######
#############################
if plot_loss 
    using PyPlot # required for launching plotting 
end

# initialising parameters
em_init = EcosystemModelMcKann(x_c = 0.4, 
                                x_p = 0.4, 
                                y_c = 1.01, 
                                y_p = 3.00,
                                R_0 = 0.4, 
                                C_0 = 0.9)

println("***************\nSimulation started for x_p = $x_p,\n noise level r= $noise,\n datasize = $datasize,\n step = $step\n***************\n")

em_true = EcosystemModelMcKann(x_c = 0.4, 
                                x_p = x_p, 
                                y_c = 2.01, 
                                y_p = 5.00, 
                                R_0 = 0.16129, 
                                C_0 = 0.5)

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
    fim = FIM_yazdani(em_true, u0_true, (tsteps[1], tsteps[end]), tsteps, em_true.p, Σ)
else
    fim = [] # FIM undefined
end

p_init = em_init.p
prob_simul = ODEProblem(em_true, u0_true, (tsteps[1], tsteps[end]), em_true.p) # needed for simulations, to only start at tsteps[1] and not 0.

res = minibatch_MLE(group_size = group_size_init,
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
                            threshold = threshold
                            )
[println(L"\hat{%$(res.p_labs[i])} = ", abs(res.p_trained[i]), L", %$(res.p_labs[i]) = ", res.p_true[i]) for i in 1:length(res.p_true)]

dict_simul = Dict{String, Any}()
p_trained = res.p_trained; p_true = res.p_true; pred = res.pred; minloss = res.minloss; nb_groups = length(res.ranges)
@pack! dict_simul = (res, x_p, p_trained, p_true, data_set, data_set_w_noise, pred, minloss, noise, step, datasize, Σ, fim, nb_groups, tsteps)
save(joinpath("results",string(_today), "$name_scenario.jld2"), dict_simul)
println("Results saved")