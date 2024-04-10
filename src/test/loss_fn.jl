#=
Testing loss functions in `loss_fn.jl`
=#

using Test
using Distributions
cd(@__DIR__)
include("../loss_fn.jl")


data_set_w_noise = rand(10, 10)
n_state_var = 5
n_tsteps = 10
pred = rand(n_state_var, n_tsteps)

p_perturbed = (r = randn(10), K = randn(10, 10))
σ_p = 0.1
scale_prior_param = 1.
ic_term = 0.
prior_param = (r = MvNormal(p_perturbed[:r], p_perturbed[:r] .* σ_p), # 
                K = MvNormal(p_perturbed[:K][:], p_perturbed[:K][:] .* σ_p))
p_init = NamedTuple([k => reshape(rand(prior_param[k]), size(p_perturbed[k])) for k in keys(prior_param)])



@testset "loss_fn_log_prior_param" begin
    l = loss_fn_log_prior_param(data_set_w_noise, 
                                p_init,
                                data_set_w_noise, 
                                [], 
                                ic_term,
                                prior_param,
                                scale_prior_param)
    @test l isa Number
end

@testset "loss_fn_lognormal_distrib" begin
    σ_noise = 0.1
    noise_distrib = MvNormal(zeros(n_state_var),  σ_noise * ones(n_state_var))
    l = loss_fn_log_prior_param(data_set_w_noise, 
                                p_init,
                                data_set_w_noise, 
                                [], 
                                ic_term,
                                prior_param,
                                scale_prior_param)
    @test l isa Number
end

# TODO: implement other loss fn