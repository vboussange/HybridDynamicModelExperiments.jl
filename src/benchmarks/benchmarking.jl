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

@model SimpleEcosystemModel

@inline function (model::SimpleEcosystemModel)(du, u, p, t)
    @unpack intinsic_growth_rate, carrying_capacity, competition, feeding, resource_conversion_efficiency = model

    T = eltype(u)
    ũ = max.(u, zero(T))

    r = intinsic_growth_rate(p, t)
    K = carrying_capacity(p, t)
    A = competition(ũ, p, t)
    ϵ = resource_conversion_efficiency(p, t)
    F = feeding(ũ, p, t)
    
    feed__pred_gains = (ϵ .* F .- F') * ũ
    du .= ũ .*(r .- A ./ K .+ feed__pred_gains)
end

## FOODWEB
const N = 3 # number of compartment

foodweb = DiGraph(N)
foodweb = DiGraph(N)
add_edge!(foodweb, 2 => 1) # C to R
add_edge!(foodweb, 3 => 2) # P to C

### ECOLOGICAL FUN
intinsic_growth_rate(p, t) = p.r
const W = adjacency_matrix(foodweb) .|> Float32
I, J, _ = findnz(adjacency_matrix(foodweb))

function feeding(u, p, t)
    @unpack H, q = p

    # handling time
    H = sparse(I, J, H, N, N)

    # attack rates
    q = sparse(I, J, q, N, N)

    return q .* W ./ (one(eltype(u)) .+ q .* H .* (W * u))
end

datasize = 60
step = 4
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6

tsteps = range(500f0, step = step, length = datasize)
tspan = (0f0, tsteps[end])

p_true = ComponentArray(H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        K₁₁ = Float32[1.0],
                        A = Float32[1.0])

u0_true = Float32[0.77, 0.060, 0.945]

mp = ModelParams(; p=p_true,
    tspan,
    u0=u0_true,
    alg,
    saveat=tsteps,
    abstol,
    reltol,
    maxiters=50_000
)

model = SimpleEcosystemModel(; mp)
using Plots
plot(simulate(model))

@btime simulate($model) # 46.589 ms (792050 allocations: 58.14 MiB)
# @code_warntype simulate(model)

# @btime intinsic_growth_rate(p_true, nothing) #48.330 ns (1 allocation: 48 bytes)
# @btime carrying_capacity(p_true, nothing) # 171.303 ns (5 allocations: 320 bytes)
# @btime competition(u0_true, p_true, nothing) # 434.256 ns (14 allocations: 960 bytes)
# @btime resource_conversion_efficiency(p_true, 0.) # 1.208 ns (0 allocations: 0 bytes) 
# @btime feeding(u0_true, p_true, nothing) # 2.292 μs (45 allocations: 3.53 KiB)