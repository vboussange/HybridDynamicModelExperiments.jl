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
(forcing::TrueTempForcing)(t) = [forcing.interpolation(t)]
ndims(forcing::TrueTempForcing) = 1

struct TrueForcing{I} <: AbstractTrueForcing # 3D true forcing
    interpolation::Vector{I}
end

function TrueForcing() 
    forcing_df = load_data_forcing()
    return TrueForcing([linear_interpolation(Dates.days.(forcing_df[:, 1]), f, extrapolation_bc = Flat()) for f in eachcol(forcing_df[:, 2:end])])
end
(forcing::TrueForcing)(t) = vcat([interp(t) for interp in forcing.interpolation]...)
ndims(forcing::TrueForcing) = length(forcing.interpolation)

struct SyntheticTempForcing{T} <: AbstractForcing
    T_max::T
    T_mean::T
end
(forcing::SyntheticTempForcing{T})(t) where T = (forcing.T_max - forcing.T_mean) * cos((T(2π) * (t - T(32))) / T(365))


struct ModelBeninca{MP, F} <: AbstractODEModel
    mp::MP
    mortality::F
end

function classic_beninca_model(mp, forcing::AbstractForcing)
    mortality(t, p) = p[[:m_A, :m_M]] .* (one(eltype(p)) + p.α[] * forcing(t))
    return ModelBeninca(mp, mortality)
end

rbf(x) = exp.(-(x.^2)) # custom activation function

function hybrid_beninca_model(mp, forcing::AbstractForcing; HlSize=5, seed=0)
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    # neural net takes forcing, and outputs mortality rates
    # of A and M
    _neural_net = Lux.Chain(
        Lux.Dense(ndims(forcing), HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, 2, x -> clamp.(x, 1f-2, 2f-2)), # output layer with 2 outputs (m_A and m_M)
    )

    p_nn, st = Lux.setup(rng, _neural_net)
    mortality_rate = StatefulLuxLayer{true}(_neural_net, nothing, st)
    mortality(t, p) = mortality_rate(forcing(t), p.p_nn)

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)

    return ModelBeninca(mp, mortality)
end


function (model::ModelBeninca)(du, u, p, t)
    T = eltype(u)
    ũ = max.(u, zero(T))

    # B₀ (cover by barnacles without crustose algae), Bᵢ (cover by barnacles
    # overgrown with crustose algae), A (coverage of crustose algae), M (coverage by
    # mussels)
    B₀, Bᵢ, A, M = ũ
    @unpack c_BR, c_AB, c_M, c_AR, m_B = p

    R = one(T) - B₀ - A - M
    m_A, m_M = model.mortality(t, p)

    du[1] = c_BR[] * (B₀ + Bᵢ) * R - c_AB[] * A * B₀ - c_M[] * M * B₀ - m_B[] * B₀ + m_A * Bᵢ
    du[2] = c_AB[] * A * B₀ - c_M[] * M * Bᵢ - m_B[] * Bᵢ - m_A * Bᵢ
    du[3] = c_AR[] * A * R + c_AB[] * A * B₀ - c_M[] * M * A - m_A * A
    du[4] = c_M[] * M * (B₀ + A) - m_M * M
end