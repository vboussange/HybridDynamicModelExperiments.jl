using SparseArrays
using ComponentArrays
using UnPack
using PiecewiseInference
import PiecewiseInference: AbstractODEModel
using Lux
import Lux: zeros32
using Random

struct HybridFuncRespModel{MP, II, F, ST} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    func_resp::F # functional response
    st::ST # neural net state
end

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
    # neural_net = Lux.Chain(Lux.Dense(3, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, 2, relu, init_weight=truncated_normal(; std=1e-4), use_bias=false))

    mlp() = Lux.Chain(Lux.Dense(1, HlSize, relu), 
                        Lux.Dense(HlSize, HlSize, relu), 
                        Lux.Dense(HlSize, HlSize, relu), 
                        Lux.Dense(HlSize, 1, relu, use_bias=false))
    _func_resp = Parallel(nothing, mlp(), mlp())
    function func_resp(u, p_nn, st)
        x = ([u[1]], [u[2]])
        y, st = _func_resp(x, p_nn, st)
        return vcat(y...)
    end


    p_nn, st = Lux.setup(rng, _func_resp)
    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridFuncRespModel(mp, I, J, func_resp, st)
end


function feed_pred_gains(m::HybridFuncRespModel, u, p)
    @unpack st, func_resp, I, J = m
    # TODO: state may need to be updated
    y, st = func_resp(u[1:2], p.p_nn, st)
    F = sparse(I, J, y, 3, 3)
    return (F - F') * u
end

struct HybridFuncRespModel2{MP, II, ST} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    neural_net::Lux.Chain # neural net
    st::ST # neural net state
end

function HybridFuncRespModel2(mp, HlSize=5, seed=0)

    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer

    I, J, _ = findnz(adjacency_matrix(foodweb))

    # neural net
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    # can only throw positive values
    # neural_net = Lux.Chain(Lux.Dense(3, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, HlSize, relu, init_weight=truncated_normal(; std=1e-4), init_bias=zeros32), 
    #                         Lux.Dense(HlSize, 2, relu, init_weight=truncated_normal(; std=1e-4), use_bias=false))

    neural_net = Lux.Chain(Lux.Dense(3, HlSize, relu), 
                            Lux.Dense(HlSize, HlSize, relu),
                            Lux.Dense(HlSize, HlSize, relu), 
                            Lux.Dense(HlSize, 3, tanh))

    p_nn, st = Lux.setup(rng, neural_net)
    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridFuncRespModel2(mp, I, J, neural_net, st)
end


function loss(m::HybridFuncRespModel2, u, p)
    @unpack st, neural_net, I, J = m
    # TODO: state may need to be updated
    y, st = neural_net(u, p.p_nn, st)
    return 2 * y .* u
end

function (model::HybridFuncRespModel2)(du, u, p, t)
    # need to clip to prevent exponential growth or negative abundances
    ũ = max.(u, zero(eltype(u)))
    ũ = min.(u, 5f0) 
    du .= ũ .* (intinsic_growth_rate(model, p, t) - loss(model, ũ, p))
    return nothing
end