#=

generating two uncorrelated time series
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

datasize = 40
step = 4
group_size = 6
noise = 0.5
x_p = 0.09
_today = today()

u0s_true = [[0.8,0.8,0.8], [0.1,1.,2.8]]
name_scenario = "3-species-model_McKann_simple_minibatch_allsp"

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
maxiters = [4000]
group_size_init = 6 + 1 # number of points in each segment for multiple shooting
continuity_term = 0. #200. is the standard value. The more the noise, the less the continuity term
ic_term = 1.

p_labs = [L"x_c", L"x_p", L"y_c", L"y_p", L"R_0", L"C_0"]

#############################
#### starting looping #######
#############################
if plot_loss 
    using PyPlot # required for launching plotting 
end

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

ode_datas_wnoise = []
for u0_true in u0s_true
    prob = ODEProblem(em_true, u0_true, tspan, em_true.p)
    sol_data = solve(prob, alg, saveat = tsteps, abstol=abstol, reltol=reltol, sensealg = sensealg)
    # using Plots
    # Plots.plot(sol_data)
    ## Ideal data
    data_set = Array(sol_data)

    # Data with noise
    std_noise = std(data_set, dims=2)[:] .* noise
    data_set_w_noise = data_set + randn(size(data_set)...)  .* std_noise
    push!(ode_datas_wnoise, data_set_w_noise)
end

ranges = MiniBatchInference._get_ranges(group_size, datasize)

@save "ode_datas_wnoise.jld2" ode_datas_wnoise tsteps ranges