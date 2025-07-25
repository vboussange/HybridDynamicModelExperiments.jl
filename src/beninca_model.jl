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
using Interpolations
DATA_PATH = joinpath(@__DIR__, "../data/beninca")


function load_data_forcing()
    forcing_df = DataFrame(CSV.File(joinpath(DATA_PATH, "forcing_merged.csv")))
    forcing_df[!, 1] = Date.(forcing_df[:, 1]) # Convert first column to Date
    forcing_df[!, 2:end] = Float32.(forcing_df[:, 2:end])

    # Normalise forcing columns (zero mean, unit variance)
    for col in names(forcing_df)[2:end]
        μ = mean(forcing_df[!, col])
        σ = std(forcing_df[!, col])
        forcing_df[!, col] = (forcing_df[!, col] .- μ) ./ σ
    end
    return forcing_df
end

# Forcings
abstract type AbstractForcing end
abstract type AbstractTrueForcing <: AbstractForcing end

struct TrueTempForcing{I} <: AbstractTrueForcing # 1D true forcing
    interpolation::I
end
function TrueTempForcing() 
    forcing_df = load_data_forcing()
    return TrueTempForcing(linear_interpolation(Dates.days.(forcing_df[:, 1]), forcing_df[:, "Temperature [oC]"], extrapolation_bc = Flat()))
end
(forcing::TrueTempForcing)(t) = forcing.interpolation(t)

struct TrueForcing{I} <: AbstractTrueForcing # 3D true forcing
    interpolation::Vector{I}
end

function TrueForcing() 
    forcing_df = load_data_forcing()
    return TrueForcing([linear_interpolation(Dates.days.(forcing_df[:, 1]), f, extrapolation_bc = Flat()) for f in eachcol(forcing_df[:, 2:end])])
end
(forcing::TrueForcing)(t) = vcat([interp(t) for interp in forcing.interpolation]...)

struct SyntheticTempForcing{T} <: AbstractForcing
    T_max::T
    T_mean::T
end

(forcing::SyntheticTempForcing{T})(t) where T = (forcing.T_max - forcing.T_mean) * cos((T(2π) * (t - T(32))) / T(365))


struct ModelBeninca{MP, F} <: AbstractODEModel
    mp::MP
    forcing::F
end

function (model::ModelBeninca)(du, u, p, t)
    T = eltype(u)
    ũ = max.(u, zero(T))

    # B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
    # overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
    # mussels)
    B₀, Bᵢ, A, M = ũ
    @unpack c_BR, c_AB, c_M, c_AR, m_B, m_A, m_M, α = p

    R = one(T) - B₀ - A - M
    F_t = one(T) + α[] * model.forcing(t)

    du[1] = c_BR[] * (B₀ + Bᵢ) * R - c_AB[] * A * B₀ - c_M[] * M * B₀ - m_B[] * B₀ + F_t * m_A[] * Bᵢ
    du[2] = c_AB[] * A * B₀ - c_M[] * M * Bᵢ - m_B[] * Bᵢ - F_t * m_A[] * Bᵢ
    du[3] = c_AR[] * A * R + c_AB[] * A * B₀ - c_M[] * M * A - F_t * m_A[] * A
    du[4] = c_M[] * M * (B₀ + A) - F_t * m_M[] * M
end

struct HybridBenincaModel{MP, F, NN} <: AbstractODEModel
    mp::MP
    forcing::F
    mortality::NN
end

rbf(x) = exp.(-(x.^2)) # custom activation function

function HybridBenincaModel(mp, forcing; HlSize=5, seed=0)
    # neural net
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    # neural net takes forcing (3 dimensional), and outputs mortality rates
    # of A and M
    _neural_net = Lux.Chain(
        Lux.Dense(3, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, 2, rbf),
    )

    p_nn, st = Lux.setup(rng, _neural_net)
    mortality_rate = StatefulLuxLayer{true}(_neural_net, nothing, st)

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridBenincaModel(mp, forcing, mortality_rate)
end

function (model::HybridBenincaModel)(du, u, p, t)
    T = eltype(u)
    ũ = max.(u, zero(T))

    # B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
    # overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
    # mussels)
    B₀, Bᵢ, A, M = ũ

    @unpack c_BR, c_AB, c_M, c_AR, m_B, p_nn = p

    R = one(T) - B₀ - A - M
    m_A, m_M = model.mortality(model.forcing(t), p_nn)

    du[1] = c_BR[] * (B₀ + Bᵢ) * R - c_AB[] * A * B₀ - c_M[] * M * B₀ - m_B[] * B₀ + m_A * Bᵢ
    du[2] = c_AB[] * A * B₀ - c_M[] * M * Bᵢ - m_B[] * Bᵢ - m_A * Bᵢ
    du[3] = c_AR[] * A * R + c_AB[] * A * B₀ - c_M[] * M * A - m_A * A
    du[4] = c_M[] * M * (B₀ + A) - m_M * M
end