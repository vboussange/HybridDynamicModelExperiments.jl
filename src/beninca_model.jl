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
DATA_PATH = joinpath(@__DIR__, "../data/beninca")


function load_data_forcing(window_size = 20)
    forcing_df = DataFrame(CSV.File(joinpath(DATA_PATH, "forcing_merged.csv")))
    forcing_df[!, 1] = Date.(forcing_df[:, 1]) # Convert first column to Date
    forcing_df[!, 2:end] = Float32.(forcing_df[:, 2:end])

    # Calculating running average
    for col in names(forcing_df)[2:end]
        forcing_df[!, col] = runmean(forcing_df[!, col],
                                    window_size,
                                                   )
    end

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

struct DummyForcing{T} <: AbstractForcing
    forcing::T
end
(forcing::DummyForcing)(t) = [forcing.forcing]
ndims(forcing::DummyForcing) = 1

struct TrueForcing{I} <: AbstractTrueForcing # 3D true forcing
    interpolation::Vector{I}
end

function TrueForcing(T::DataType, 
                    vars = ["Temperature [oC]", "Wave height [m]", "Wind run [km/day]"], 
                    window_size = 20) 
    forcing_df = load_data_forcing(window_size)
    return TrueForcing([linear_interpolation(Dates.days.(forcing_df[:, 1]), forcing_df[:, v] .|> T, extrapolation_bc = Flat()) for v in vars])
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

function hybrid_beninca_model(mp, forcing::AbstractForcing; HlSize=20, seed=0)
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    # neural net takes forcing, and outputs mortality rates
    # of A and M
    _neural_net = Lux.Chain(
        Lux.Dense(ndims(forcing), HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        Lux.Dense(HlSize, HlSize, relu), 
        # Lux.Dense(HlSize, HlSize, relu), 
        # # Lux.Dense(HlSize, 2, x -> 2f-2 .* rbf.(x)), # output layer with 2 outputs (m_A and m_M)
        Lux.Dense(HlSize,
                2, 
                # init_weight = glorot_uniform(; gain=1e-6)
                ) # output layer with 2 outputs (m_A and m_M) with bijector
    )

    p_nn, st = Lux.setup(rng, _neural_net)
    mortality_rate = StatefulLuxLayer{true}(_neural_net, nothing, st)
    last_activ_fun = inverse(bijector(Uniform(1f-4, 5f-2)))
    # last_activ_fun = x -> clamp.(x, eltype(x)(1e-4), eltype(x)(5e-2))

    function mortality(t, p) 
        x = mortality_rate(forcing(t), p.p_nn) # TODO: we could look back in the forcing history
        x = last_activ_fun(x) # apply inverse bijector to the output
        return x
    end

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)

    return ModelBeninca(mp, mortality)
end


function (model::ModelBeninca)(du, u, p, t)
    T = eltype(u)
    ũ = clamp.(u, zero(T), one(T)) # clamp to [0, 1] to avoid negative or too large values

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