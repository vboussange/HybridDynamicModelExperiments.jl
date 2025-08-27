#=
5 species model.
=#
using SparseArrays
using ComponentArrays

struct Model5SP{II,JJ} <: AbstractEcosystemModel
    I::II
    J::JJ
end

name(::Model5SP) = "Model5SP"

function Model5SP()
    ## FOODWEB
    foodweb = DiGraph(5)
    add_edge!(foodweb, 2 => 1) # C1 to R1
    add_edge!(foodweb, 5 => 4) # C2 to R2
    add_edge!(foodweb, 3 => 2) # P to C1
    add_edge!(foodweb, 3 => 5) # P to C2

    I, J, _ = findnz(adjacency_matrix(foodweb))

    Model5SP(I, J)
end

function competition(::Model5SP, u, p)
    @unpack A = p
    T = eltype(A)
    Au = vcat(A[1] * u[1], zeros(T,2), A[2] * u[4], zero(T))
    return Au
end

function create_sparse_matrices(model::Model5SP, p)
    @unpack I, J = model
    @unpack ω, H, q = p

    # creating foodweb
    OT = eltype(ω)
    Warr = sparse(I, J, vcat(one(OT), ω[], one(OT) - ω[], one(OT)), 5, 5)

    # handling time
    Harr = sparse(I, J, H, 5, 5)

    # attack rates
    qarr = sparse(I, J, q, 5, 5)
    
    return Warr, Harr, qarr
end

function get_metadata(::Model5SP)
    species_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
    node_labels = ["R1", 
                    "C1", 
                    "P1", 
                    "R2", 
                    "C2"]

    pos = Dict(0 => [0, 0], 1 => [1, 1], 2 => [2, 2], 3 => [4, 0], 4 => [3, 1],)
    return (; species_colors, node_labels, pos)
end

