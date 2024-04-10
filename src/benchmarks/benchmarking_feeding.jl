#=
Ecological functions to define the 3 species model

=#
using SparseArrays
using Graphs
using ComponentArrays
using UnPack
using BenchmarkTools
using OrdinaryDiffEq
using ParametricModels

## FOODWEB
const N = 3 # number of compartment

foodweb = DiGraph(N)
foodweb = DiGraph(N)
add_edge!(foodweb, 2 => 1) # C to R
add_edge!(foodweb, 3 => 2) # P to C

### ECOLOGICAL FUN
const W = adjacency_matrix(foodweb) .|> Float32
const I, J, _ = findnz(adjacency_matrix(foodweb))

function feeding(u, p, t)
    @unpack H, q = p

    # handling time
    Harr = sparse(I, J, H, 3, 3)

    # attack rates
    qarr = sparse(I, J, q, 3, 3)

    return  inv.((inv.(qarr)  .+ Harr .* (W * u)))
end

function feeding2(u, p, t)
    @unpack H, q = p

    # handling time
    Harr = sparse(I, J, H, 3, 3)

    # attack rates
    qarr = sparse(I, J, q, 3, 3)

    return qarr .* W ./ (one(eltype(u)) .+ qarr .* Harr .* (W * u))
end

p_true = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K₁₁ = Float32[1.0],
                        A = Float32[1.0])

u0_true = Float32[0.77, 0.060, 0.945]

using Test
@test all(feeding(u0_true, p_true, nothing) .== feeding2(u0_true, p_true, nothing))

@btime feeding(u0_true, p_true, nothing) # 2.292 μs (45 allocations: 3.53 KiB)

@btime feeding2(u0_true, p_true, nothing) # 2.292 μs (45 allocations: 3.53 KiB)