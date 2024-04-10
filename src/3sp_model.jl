using SparseArrays
using ComponentArrays
using ParametricModels
using UnPack

abstract type Abstract3SPModel <: AbstractModel end

function (model::Abstract3SPModel)(du, u, p, t)
    @unpack A = p
    r = intinsic_growth_rate(model, p, t)
    ũ = max.(u, zero(eltype(u)))

    F = feeding(model, ũ, p)
    Aarr = competition(model, A, ũ[1])

    feed_pred_gains = (F .- F') * ũ
    du .= ũ .* (r .- Aarr .+ feed_pred_gains)
end

intinsic_growth_rate(::Abstract3SPModel, p, t) = p.r

function competition(::Abstract3SPModel, A, u1)
    T = eltype(u1)
    return [A[1] * u1; zeros(T, 2)]
end

function feeding(m::Abstract3SPModel, u, p)
    @unpack I, J = m
    # bottleneck 
    Warr, Harr, qarr = create_sparse_matrices(m, I, J, p)
    return qarr .* Warr ./ (one(eltype(u)) .+ qarr .* Harr .* (Warr * u))
end


struct SimpleEcosystemModel3SP{MP,II,JJ} <: Abstract3SPModel
    mp::MP
    I::II
    J::JJ
end

function SimpleEcosystemModel3SP(mp)
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))
    SimpleEcosystemModel3SP(mp, I, J)
end

function create_sparse_matrices(::Abstract3SPModel, I, J, p)
    @unpack H, q = p
    OT = eltype(H)
    Warr = sparse(I, J, ones(OT, 2), 3, 3)
    Harr = sparse(I, J, H, 3, 3)
    qarr = sparse(I, J, q, 3, 3)
    return Warr, Harr, qarr
end

struct SimpleEcosystemModelOmniv3SP{MP,II,JJ} <: Abstract3SPModel
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


function get_metadata(::Abstract3SPModel)
    species_colors = ["tab:red", "tab:green", "tab:blue"]
    pos = Dict(0 => [0, 0], 1 => [0.2, 1], 2 => [0, 2])
    node_labels = ["Resource", "Consumer", "Prey"]
    return (; species_colors, pos, node_labels)
end