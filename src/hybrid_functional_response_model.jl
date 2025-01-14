using SparseArrays
using ComponentArrays
using UnPack
using PiecewiseInference
import PiecewiseInference: AbstractODEModel
using Lux
import Lux: zeros32
using Random

struct HybridFuncRespModel{MP, II, ST} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    neural_net::Lux.Chain # neural net
    st::ST # neural net state
end

rbf(x) = exp.(-(x.^2)) # custom activation function

function HybridFuncRespModel(mp, HlSize=5, seed=0)

    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))


    # neural net
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    # can only throw positive values
    neural_net = Lux.Chain(Lux.Dense(3, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
                            Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
                            Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
                            Lux.Dense(HlSize, 2, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32, use_bias=false))

    p_nn, st = Lux.setup(rng, neural_net)
    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridFuncRespModel(mp, neural_net, st, I, J)
end


function feed_pred_gains(m::HybridFuncRespModel, u, p)
    @unpack st, neural_net, I, J = m
    # TODO: state may need to be updated
    y, st = neural_net(u, p.p_nn, st)
    F = sparse(I, J, y, 3, 3)
    return (F - F') * u
end