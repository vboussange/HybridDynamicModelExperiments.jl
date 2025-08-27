#=
Implementation of the model presented in:
https://www.pnas.org/doi/10.1073/pnas.1421968112
=#
using PiecewiseInference
using PiecewiseInference: AbstractODEModel
using ComponentArrays
using UnPack
using Lux
import Lux: zeros32
using Random
using NNlib
using Interpolations: linear_interpolation, Flat
using RollingFunctions: runmean

struct tDepBenincaModel{MP, F} <: AbstractODEModel
    mp::MP
    fun::F
end

function tdep_beninca_model(mp, forcing::AbstractForcing; HlSize=20, seed=0)
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    # neural net takes forcing, and outputs mortality rates
    # of A and M
    _neural_net = Lux.Chain(
        Lux.Dense(ndims(forcing), HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        # # Lux.Dense(HlSize, 2, x -> 2f-2 .* rbf.(x)), # output layer with 2 outputs (m_A and m_M)
        Lux.Dense(HlSize,
                7, 
                # init_weight = glorot_uniform(; gain=1e-6)
                ) # output layer with 2 outputs (m_A and m_M) with bijector
    )

    p_nn, st = Lux.setup(rng, _neural_net)
    mortality_rate = StatefulLuxLayer{true}(_neural_net, nothing, st)
    last_activ_fun = inverse(bijector(Uniform(1f-4, 5f-2)))
    # last_activ_fun = x -> clamp.(x, eltype(x)(1e-4), eltype(x)(5e-2))

    function parameters(t, p) 
        x = mortality_rate(forcing(t), p.p_nn) # TODO: we could look back in the forcing history
        x = last_activ_fun(x) # apply inverse bijector to the output
        return x
    end

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)

    return tDepBenincaModel(mp, parameters)
end

function (model::tDepBenincaModel)(du, u, p, t)
    T = eltype(u)
    ũ = clamp.(u, zero(T), one(T)) # clamp to [0, 1] to avoid negative or too large values

    # B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
    # overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
    # mussels)
    B₀, Bᵢ, A, M = ũ
    c_BR, c_AB, c_M, c_AR, m_B, m_A, m_M  = model.fun(t, p)

    R = one(T) - B₀ - A - M

    du[1] = c_BR * (B₀ + Bᵢ) * R - c_AB * A * B₀ - c_M * M * B₀ - m_B * B₀ + m_A * Bᵢ
    du[2] = c_AB * A * B₀ - c_M * M * Bᵢ - m_B * Bᵢ - m_A * Bᵢ
    du[3] = c_AR * A * R + c_AB * A * B₀ - c_M * M * A - m_A * A
    du[4] = c_M * M * (B₀ + A) - m_M * M
end