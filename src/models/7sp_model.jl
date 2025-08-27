#=
7 species model
=#
using SparseArrays
using ComponentArrays

struct Model7SP{II,JJ} <: AbstractODEModel
    I::II
    J::JJ
end

function Model7SP()
    ## FOODWEB
    foodweb = DiGraph(7)
    add_edge!(foodweb, 2 => 1) # C1 to R1
    add_edge!(foodweb, 5 => 4) # C2 to R2
    add_edge!(foodweb, 3 => 2) # P1 to C1
    add_edge!(foodweb, 3 => 5) # P1 to C2
    add_edge!(foodweb, 6 => 3) # P2 to P1
    add_edge!(foodweb, 7 => 6) # P3 to P2

    I, J, _ = findnz(adjacency_matrix(foodweb))

    Model7SP(I, J)
end

function competition(::Model7SP, u, p)
    @unpack A = p
    T = eltype(A)
    Au = vcat(A[1] * u[1], zeros(T,2), A[2] * u[4], zeros(T,3))
    return Au
end

function create_sparse_matrices(model::Model7SP, p)
    @unpack I, J = model
    @unpack ω, H, q = p

    # creating foodweb
    OT = eltype(ω)
    Warr = sparse(I, J, vcat(one(OT), ω, one(OT), one(OT) .- ω, ones(OT, 2)), 7, 7)

    # handling time
    Harr = sparse(I, J, H, 7, 7)

    # attack rates
    qarr = sparse(I, J, q, 7, 7)

    return Warr, Harr, qarr
end

function get_metadata(::Model7SP)
    species_colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple", "tab:pink", "tab:gray"]
    node_labels = ["R1", 
                    "C1", 
                    "P1", 
                    "R2", 
                    "C2",
                    "P2",
                    "P3"]
    
    pos = Dict(0 => [0, 0], 1 => [1, 1], 2 => [2, 2], 3 => [4, 0], 4 => [3, 1], 5 => [2, 3], 6 => [2, 4])
    return (; species_colors, node_labels, pos)
end

## PLOTTING


