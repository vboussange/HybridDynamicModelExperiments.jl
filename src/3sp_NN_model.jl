using SparseArrays
using Lux
using Random
const rng = Random.default_rng()
using ParametricModels
const Period = 2 * pi / 600 * 5

abstract type Abstract3SPrDepModel <: Abstract3SPModel end

@inline function (model::Abstract3SPrDepModel)(du, u, p, t)
    @unpack A, K₁₁ = p

    T = eltype(u)
    ũ = max.(u, zero(T))

    r = intinsic_growth_rate(model, p, t)
    F = feeding(model, ũ, p)
    
    feed_pred_gains = (F .- F') * ũ
    du[1] = ũ[1] * (r[1] - A[1] / K₁₁[1] * ũ[1] .+ feed_pred_gains[1])
    du[2:end] .= ũ[2:end] .*(r[2:end] .+ feed_pred_gains[2:end])
end

Base.@kwdef struct HybridEcosystemModel{MP,II,JJ,ST} <: Abstract3SPrDepModel
    mp::MP
    I::II
    J::JJ
    st::ST
end

# Multilayer FeedForward
const HlSize = 5
rbf(x) = exp.(-(x.^2)) # custom activation function

neural_net = Lux.Chain(Lux.Dense(1,HlSize,rbf), 
                        Lux.Dense(HlSize,HlSize, rbf), 
                        Lux.Dense(HlSize,HlSize, rbf), 
                        Lux.Dense(HlSize, 1, rbf))

growth_rate_resource_nn(p_nn, st, water) = neural_net([water], p_nn, st)[1]

function intinsic_growth_rate(m::HybridEcosystemModel, p, t)
    st = m.st
    return [growth_rate_resource_nn(p.p_nn, st, water_availability(t)); p.r]
end

function HybridEcosystemModel(mp)
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1) # C to R
    add_edge!(foodweb, 3 => 2) # P to C

    I, J, _ = findnz(adjacency_matrix(foodweb))

    _, st = Lux.setup(rng, neural_net)

    HybridEcosystemModel(mp, I, J, st)
end

Base.@kwdef struct SimpleEcosystemModel3SPStar{MP,II,JJ} <: Abstract3SPModel
    mp::MP
    I::II
    J::JJ
end

function SimpleEcosystemModel3SPStar(mp)
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1) # C to R
    add_edge!(foodweb, 3 => 2) # P to C

    I, J, _ = findnz(adjacency_matrix(foodweb))

    SimpleEcosystemModel3SPStar(mp, I, J)
end

water_availability(t::T) where T = sin.(convert(T, Period) * t)

growth_rate_resource(p, water) = p.r[1] * exp(-0.5*(water)^2 / p.s[1]^2)

intinsic_growth_rate(::SimpleEcosystemModel3SPStar, p, t) = [growth_rate_resource(p, water_availability(t)); p.r[2:end]]

# we need to redefine the simulate fn for `HybridEcosystemModel`, because nested component Arrays are incompatible with merge, so we bypass this utility
import ParametricModels.simulate
function simulate(m::HybridEcosystemModel; u0 = nothing, tspan=nothing, p, alg = nothing, kwargs...)
    isnothing(u0) ? u0 = ParametricModels.get_u0(m) : nothing
    isnothing(tspan) ? tspan = ParametricModels.get_tspan(m) : nothing
    isnothing(alg) ? alg = ParametricModels.get_alg(m) : nothing
    prob = ParametricModels.get_prob(m, u0, tspan, p)
    # kwargs erases get_kwargs(m)
    sol = solve(prob, alg; ParametricModels.get_kwargs(m)..., kwargs...)
    return sol
end