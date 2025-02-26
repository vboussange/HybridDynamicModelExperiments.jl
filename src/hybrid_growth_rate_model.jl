#=
3-species model incorporating a neural network for resource species growth rate,
and a varying growth rate model influenced by environmental factors.
=#

using SparseArrays
using Lux
using Random
using PiecewiseInference
import PiecewiseInference: AbstractODEModel
const Period = 2 * pi / 600 * 5
using Random

struct Model3SPVaryingGrowthRate{MP, II} <: AbstractModel3SP
    mp::MP
    I::II
    J::II
end
water_availability(t::T) where T = sin.(convert(T, Period) * t)
growth_rate_resource(p, water) = p.r[1] * exp(-0.5*(water)^2 / p.s[1]^2)
intinsic_growth_rate(::Model3SPVaryingGrowthRate, p, t) = [growth_rate_resource(p, water_availability(t)); p.r[2:end]]


struct HybridGrowthRateModel{MP, II, G} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    growth_rate::G # neural net
end

rbf(x) = exp.(-(x.^2)) # custom activation function

function HybridGrowthRateModel(mp; HlSize=5, seed=0)
    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))

    # neural net
    rng = TaskLocalRNG()
    Random.seed!(rng, seed)

    _neural_net = Lux.Chain(Lux.Dense(1, HlSize, rbf), 
                            Lux.Dense(HlSize, HlSize, rbf), 
                            Lux.Dense(HlSize, HlSize, rbf), 
                            Lux.Dense(HlSize, 1))

    p_nn, st = Lux.setup(rng, _neural_net)
    growth_rate = StatefulLuxLayer{true}(_neural_net, nothing, st)

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridGrowthRateModel(mp, I, J, growth_rate)
end

function intinsic_growth_rate(m::HybridGrowthRateModel, p, t)
    p_nn = p.p_nn
    r1 = m.growth_rate([water_availability(t)], p_nn)[1]
    return [r1; p.r]
end