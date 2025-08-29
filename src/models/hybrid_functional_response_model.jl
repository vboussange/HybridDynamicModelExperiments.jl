#=
3 species model with a neural network to account for functional response.
=#

using SparseArrays
using ComponentArrays
using UnPack
using Lux
import Lux: zeros32
using Random
using NNlib

struct HybridFuncRespModel{II} <: AbstractModel3SP
    I::II # foodweb row index
    J::II # foodweb col index
end

function HybridFuncRespModel()
    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))
    HybridFuncRespModel(I, J)
end

function (model::HybridFuncRespModel)(components, u, ps, t)
    p = components.parameters(ps.parameters)
    ũ = max.(u, zero(eltype(u)))
    du = ũ .* (intinsic_growth_rate(model, p, t) .- competition(model, ũ, p) .+ feed_pred_gains(model, components, ũ, ps))
    return du
end

function feeding(m::HybridFuncRespModel, components, u, ps)
    @unpack I, J = m
    y = components.functional_response(u[1:2], ps.functional_response)
    F = sparse(I, J, y, 3, 3)
    return F
end

function feed_pred_gains(m::HybridFuncRespModel, components, u, ps)
    F = feeding(m, components, u, ps)
    return  (F .- F') * u
end

nameof(::HybridFuncRespModel) = "HybridFuncRespModel"