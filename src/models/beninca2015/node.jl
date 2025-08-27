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

struct NODE{MP, F} <: AbstractODEModel
    mp::MP
    fun::F
end

function node(mp; HlSize=20, seed=0)
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    # neural net takes forcing, and outputs mortality rates
    # of A and M
    _neural_net = Lux.Chain(
        Lux.Dense(4, HlSize, softplus), 
        Lux.Dense(HlSize, HlSize, softplus), 
        Lux.Dense(HlSize, HlSize, softplus), 
        Lux.Dense(HlSize, HlSize, softplus), 
        # # Lux.Dense(HlSize, 2, x -> 2f-2 .* rbf.(x)), # output layer with 2 outputs (m_A and m_M)
        Lux.Dense(HlSize,
                4, 
                tanh
                # init_weight = glorot_uniform(; gain=1e-6)
                ) # output layer with 2 outputs (m_A and m_M) with bijector
    )

    p_nn, st = Lux.setup(rng, _neural_net)
    rates = StatefulLuxLayer{true}(_neural_net, nothing, st)

    mp = remake(mp; p = ComponentVector(p_nn))

    return NODE(mp, rates)
end

function (model::NODE)(du, u, p, t)
    T = eltype(u)
    ũ = clamp.(u, zero(T), one(T)) # clamp to [0, 1] to avoid negative or too large values

    # B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
    # overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
    # mussels)
    du .= model.fun(ũ, p)
end