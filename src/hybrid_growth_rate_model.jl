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


struct HybridGrowthRateModel{MP, II, ST} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    neural_net::Lux.Chain # neural net
    st::ST # neural net state
end

rbf(x) = exp.(-(x.^2)) # custom activation function

function HybridGrowthRateModel(mp, HlSize=5, seed=0)
    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))

    # neural net
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    # can only throw positive values
    neural_net = Lux.Chain(Lux.Dense(1, HlSize, rbf), 
                            Lux.Dense(HlSize, HlSize, rbf), 
                            Lux.Dense(HlSize, HlSize, rbf), 
                            Lux.Dense(HlSize, 1))

    p_nn, st = Lux.setup(rng, neural_net)
    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridGrowthRateModel(mp, I, J, neural_net, st)
end

function intinsic_growth_rate(m::HybridGrowthRateModel, p, t)
    st = m.st
    p_nn = p.p_nn
    growth_rate_resource_nn = m.neural_net([water_availability(t)], p_nn, st)[1]
    return [growth_rate_resource_nn; p.r]
end