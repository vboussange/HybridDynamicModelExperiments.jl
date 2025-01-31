using SparseArrays
using ComponentArrays
using UnPack
using PiecewiseInference
import PiecewiseInference: AbstractODEModel
using Lux
import Lux: zeros32
using Random
using NNlib

"""
    HybridFuncRespModel(mp, HlSize=5, seed=0)

This model is a generalization of the HybridFuncRespModel model, which does not assume a specific foodweb structure.
"""
struct HybridFuncRespModel{MP, II, F} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    func_resp::F # functional response
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
    # version 1: no separate neural net for each species
    # neural_net = Lux.Chain(Lux.Dense(2, HlSize, relu), 
    #                     Lux.Dense(HlSize, HlSize, relu), 
    #                     Lux.Dense(HlSize, HlSize, relu), 
    #                     Lux.Dense(HlSize, 2, relu, use_bias=false))
    # p_nn, st = Lux.setup(rng, neural_net)
    # func_resp = StatefulLuxLayer{true}(neural_net, nothing, st)
    
    # version 2: separate neural net for each species 
    last_activ_fun = inverse(bijector(Uniform(0f0, Inf)))
    mlp() = Lux.Chain(Lux.Dense(1, HlSize, tanh),
                        Lux.Dense(HlSize, HlSize, tanh), 
                        Lux.Dense(HlSize, HlSize, tanh), 
                        Lux.Dense(HlSize, 1))

    # TODO: the use of Parallel may be an overkill, instead we may want a tuple of neural nets to be unravelled
    # This may be more efficient
    _neural_net = Parallel(nothing, mlp(), mlp())
    p_nn, st = Lux.setup(rng, _neural_net)
    neural_net = StatefulLuxLayer{true}(_neural_net, nothing, st)

    function func_resp(u, p_nn)
        x = ([u[1]], [u[2]])
        y = neural_net(x, p_nn)
        return last_activ_fun(vcat(y...))
    end

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridFuncRespModel(mp, I, J, func_resp)
end

function feeding(m::HybridFuncRespModel, u, p)
    @unpack func_resp, I, J = m
    y = func_resp(u[1:2], p.p_nn)
    F = sparse(I, J, y, 3, 3)
    return F
end


"""
    HybridFuncRespModel2

This model is a generalization of the HybridFuncRespModel model, which does not assume a specific foodweb structure.
"""
struct HybridFuncRespModel2{MP, II, F} <: AbstractModel3SP
    mp::MP # model parameters
    I::II # foodweb row index
    J::II # foodweb col index
    func_resp::F # functional response
end

function HybridFuncRespModel2(mp, HlSize=5, seed=0)

    # foodweb
    foodweb = DiGraph(3)
    add_edge!(foodweb, 2 => 1)  # Consumer to Resource
    add_edge!(foodweb, 3 => 2)  # Predator to Consumer
    add_edge!(foodweb, 3 => 1)  # Predator to Resource

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
                            Lux.Dense(HlSize, 3, relu, use_bias=false))

    p_nn, st = Lux.setup(rng, neural_net)
    func_resp = StatefulLuxLayer{true}(neural_net, nothing, st)

    p = ComponentArray(mp.p; p_nn)
    mp = remake(mp; p)
    HybridFuncRespModel2(mp, I, J, func_resp)
end

function feeding(m::HybridFuncRespModel2, u, p)
    @unpack func_resp, I, J = m
    y = func_resp(u, p.p_nn)
    F = sparse(I, J, y, 3, 3)
    return F
end

# function loss(m::HybridFuncRespModel2, u, p)
#     @unpack st, neural_net, I, J = m
#     # TODO: state may need to be updated
#     y, st = neural_net(u, p.p_nn, st)
#     return 2 * y .* u
# end

# function (model::HybridFuncRespModel2)(du, u, p, t)
#     # need to clip to prevent exponential growth or negative abundances
#     ũ = max.(u, zero(eltype(u)))
#     ũ = min.(u, 5f0) 
#     du .= ũ .* (intinsic_growth_rate(model, p, t) - loss(model, ũ, p))
#     return nothing
# end