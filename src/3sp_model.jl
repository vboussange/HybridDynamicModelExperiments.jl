#=
3 species model.
=#

using SparseArrays
using ComponentArrays
using UnPack
using PiecewiseInference
using PiecewiseInference: AbstractODEModel
import Graphs: DiGraph, add_edge!, adjacency_matrix

abstract type AbstractEcosystemModel <: AbstractODEModel end

abstract type AbstractModel3SP <: AbstractEcosystemModel end

function (::Type{T})(mp) where T <: AbstractModel3SP 
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))
    T(mp, I, J)
end

function (model::AbstractEcosystemModel)(du, u, p, t)
    ũ = max.(u, zero(eltype(u)))
    du .= ũ .* (intinsic_growth_rate(model, p, t) .- competition(model, ũ, p) .+ feed_pred_gains(model, ũ, p))
    return nothing
end

intinsic_growth_rate(::AbstractEcosystemModel, p, t) = p.r

function competition(::AbstractModel3SP, u, p)
    @unpack A = p
    T = eltype(u)
    return [A[1] * u[1]; zeros(T, 2)]
end

function feeding(m::AbstractModel3SP, u, p)
    Warr, Harr, qarr = create_sparse_matrices(m, p)
    return qarr .* Warr ./ (one(eltype(u)) .+ qarr .* Harr .* (Warr * u))
end

function feed_pred_gains(model::AbstractModel3SP, u, p)
    F = feeding(model, u, p)
    return  (F .- F') * u
end

struct Model3SP{MP,II,JJ} <: AbstractModel3SP
    mp::MP
    I::II
    J::JJ
end

function create_sparse_matrices(m::AbstractModel3SP, p)
    @unpack I, J = m
    @unpack H, q = p
    OT = eltype(H)
    Warr = sparse(I, J, ones(OT, 2), 3, 3)
    Harr = sparse(I, J, H, 3, 3)
    qarr = sparse(I, J, q, 3, 3)
    return Warr, Harr, qarr
end

struct SimpleEcosystemModelOmniv3SP{MP,II,JJ} <: AbstractModel3SP
    mp::MP
    I::II
    J::JJ
end

function SimpleEcosystemModelOmniv3SP(mp)
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer
    add_edge!(foodweb, 3 => 1)  # Predator to Resource

    I, J, _ = findnz(adjacency_matrix(foodweb))
    SimpleEcosystemModelOmniv3SP(mp, I, J)
end

function create_sparse_matrices(::SimpleEcosystemModelOmniv3SP, I, J, p)
    @unpack ω, H, q = p
    OT = eltype(ω)
    Warr = sparse(I, J, vcat(one(OT), ω, one(OT) .- ω), 3, 3) # preference coefficients
    Harr = sparse(I, J, H, 3, 3) # Handling times
    qarr = sparse(I, J, q, 3, 3) # attack rates
    return Warr, Harr, qarr
end


function get_metadata(::AbstractModel3SP)
    species_colors = ["tab:red", "tab:green", "tab:blue"]
    pos = Dict(0 => [0, 0], 1 => [0.2, 1], 2 => [0, 2])
    node_labels = ["Resource", "Consumer", "Prey"]
    return (; species_colors, pos, node_labels)
end