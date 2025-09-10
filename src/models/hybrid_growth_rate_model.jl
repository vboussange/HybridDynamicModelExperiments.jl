#=
3-species model incorporating a neural network for resource species growth rate,
and a varying growth rate model influenced by environmental factors.
=#

using SparseArrays
using Lux
using Random
const Period = 2 * pi / 600 * 5
using Random

struct VaryingGrowthRateModel{II} <: AbstractModel3SP
    I::II
    J::II
end
water_availability(t::T) where T = sin.(convert(T, Period) * t)
growth_rate_resource(p, water::T) where T = p.r[1] * exp(-T(0.5)*(water)^2 / p.s[1]^2)
intinsic_growth_rate(::VaryingGrowthRateModel, p, t) = [growth_rate_resource(p, water_availability(t)); p.r[2:end]]


struct HybridGrowthRateModel{II} <: AbstractModel3SP
    I::II # foodweb row index
    J::II # foodweb col index
end

function (model::HybridGrowthRateModel)(components, u, ps, t)
    p = components.parameters(ps.parameters)
    ũ = max.(u, zero(eltype(u)))
    du = ũ .* (intinsic_growth_rate(model, components, ps, p, t) .- competition(model, ũ, p) .+ feed_pred_gains(model, ũ, p))
    return du
end

function intinsic_growth_rate(::HybridGrowthRateModel, components, ps, p, t)
    # TODO: this sometimes fails with
    # to reproduce, check out first instance of gridsearch hybrid growth rate
    #=
    ERROR: MethodError: (::ReverseDiff.ForwardOptimize{typeof(+)})(::ReverseDiff.TrackedReal{…}, ::ReverseDiff.TrackedReal{…}) is ambiguous.

    Candidates:
    (self::ReverseDiff.ForwardOptimize{F})(x::Real, t::ReverseDiff.TrackedReal{V, D}) where {F, V, D}
        @ ReverseDiff ~/.julia/packages/ReverseDiff/rKZaG/src/macros.jl:109
    (self::ReverseDiff.ForwardOptimize{F})(t::ReverseDiff.TrackedReal{V, D}, x::Real) where {F, V, D}
        @ ReverseDiff ~/.julia/packages/ReverseDiff/rKZaG/src/macros.jl:121

    Possible fix, define
    (::ReverseDiff.ForwardOptimize{…})(::ReverseDiff.TrackedReal{…}, ::ReverseDiff.TrackedReal{…}) where {…}


    # could be due to mixed precision; change tsteps with luxtype (although should already be luxtype with fmap)
    =#
    r1 = components.growth_rate([water_availability(t)], ps.growth_rate)[1]
    return [r1; p.r]
end

nameof(::HybridGrowthRateModel) = "HybridGrowthRateModel"