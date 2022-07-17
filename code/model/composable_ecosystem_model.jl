#= 
constructs a composable ecosystem model 
to ease the constructrion of different variants
with different functional response 
for higher trophic levels
=#
using DiffEqFlux, Flux
using OrdinaryDiffEq
using MiniBatchInference

"""
    EcosystemModel(f1, f2, p1, p2, ds)

This model is inspired from Hastings 1991.
# Arguments
- `f1`, `f2`: functions representing functional response, with 
`(x,p)` as arguments
- `p1`, `p2`: arguments of functional responses
- `ds`: death rates
"""
struct EcosystemModel{P,F1,F2,F3} <: Function
    p::P # parameter vector
    F1::F1 # re function
    F2::F2 # re function
    F3::F3 # re function
end

function EcosystemModel(f1, f2, f3, p1, p2, p3, ds)
    @assert length(p1) + length(p2) + length(p3) + length(ds) == length(f1) + length(f2) + length(f3) + 2
    p = [ds; p1; p2; p3]
    EcosystemModel{typeof(p),typeof(f1),typeof(f2),typeof(f3)}(p,f1,f2,f3)
end

# default behavior : only predator relationships
EcosystemModel(f1, f2, p1, p2, ds) = EcosystemModel(f1, f2, NullResp(), p1, p2, eltype(p1)[], ds)

Flux.@functor EcosystemModel (p,)

function (em::EcosystemModel)(du, u, p, t)
    ũ = max.(u, 0.)
    ds = p[1:2]
    _p1 = p[3:2+length(em.F1)]
    _p2 = p[3+length(em.F1):2+length(em.F1)+length(em.F2)]
    _p3 = p[3+length(em.F1)+length(em.F2):end]
    f_u1 = em.F1(ũ[1], ũ[2] ,_p1)
    f_u2 = em.F2(ũ[2], ũ[3], _p2)
    f_u3 = em.F3(ũ[1], ũ[3], _p3)
    du[1] = ũ[1] * (1. - ũ[1]) - f_u1 * ũ[2] - f_u3 * ũ[3]
    du[2] = f_u1 * ũ[2] - f_u2 * ũ[3] - ds[1] * ũ[2]
    du[3] = f_u2 * ũ[3] + f_u3 * ũ[3] - ds[2] * ũ[3];
end

# overloading to have the behavior f(x,y,p) = f(x,p) when f(x,y,p) not defined
abstract type FuncResp{T} <: ParamFun{T} end
(f::FuncResp)(x, y, p) = f(x, p) 

# type I functional response
struct f_I <: FuncResp{1} end
(f::f_I)(x, p) = p[1] * x

# type II functional response
struct f_II <: FuncResp{2} end
(f::f_II)(x, p) = p[1] * x / (1. + p[2] * x)

# type II functional response
struct f_III <: FuncResp{2} end
(f::f_III)(x, p) = p[1] * x^2 / (1. + p[2] * x^2)

# Bedington DeAngelis functional response
struct f_BedAng <: FuncResp{3} end
(f::f_BedAng)(x, y, p) = p[1] * x / (1. + p[2] * x + p[3] * y)

# Null response
struct NullResp <: FuncResp{0} end
(f::NullResp)(x, p) = zero(eltype(x))




##########################################################################


"""
    EcosystemModelOmnivory(;x_c, x_p, y_c, y_pr, y_pc, R_0, R_02, C_0, ω)

This model is inspired from McCann 1997.
"""
struct EcosystemModelOmnivory{P} <: Function
    p::P # parameter vector
end

function EcosystemModelOmnivory(;x_c, x_p, y_c, y_pr, y_pc, R_0, R_02, C_0, ω)
    p = [x_c, x_p, y_c, y_pr, y_pc, R_0, R_02, C_0, ω]
    EcosystemModelOmnivory{typeof(p)}(p)
end

function (em::EcosystemModelOmnivory)(du, u, p, t)
    ũ = max.(u, 0.)
    p̃ = abs.(p)
    x_c, x_p, y_c, y_pr, y_pc, R_0, R_02, C_0, ω = p̃
    R, C, P = ũ
    du[1] = R * (1. - R) - x_c * y_c * C * R / (R + R_0) - ω * x_p * y_pr * P * R / (R_02 + (1. - ω) * C + ω * R) #R
    du[2] = - x_c * C * ( 1. - y_c * R / (R + R_0)) - (1. - ω) * x_p * y_pc * P * C / (ω * R + (1. - ω) * C + C_0) #C
    du[3] = - x_p * P + (1. - ω) * x_p * y_pc * C * P / (ω * R + (1. - ω) * C + C_0) + ω * x_p * y_pr * P * R / (ω * R + (1. - ω) * C + R_02) #P
end


"""
    EcosystemModelMcKann(;x_c, x_p, y_c, y_p, R_0, C_0)

This model is inspired from McCann 1994. Similar to EcosystemModelOmnivory, 
but without monivory.
"""
struct EcosystemModelMcKann{P} <: Function
    p::P # parameter vector
end
function EcosystemModelMcKann(;x_c, x_p, y_c, y_p, R_0, C_0)
    p = [x_c, x_p, y_c, y_p, R_0, C_0]
    EcosystemModelMcKann{typeof(p)}(p)
end
function (em::EcosystemModelMcKann)(du, u, p, t) 
    ũ = max.(u, 0.)
    p̃ = abs.(p)
    x_c, x_p, y_c, y_p, R_0, C_0 = p̃
    R, C, P = ũ
    du[1] = R * (1. - R) - x_c * y_c * C * R / (R + R_0) #R
    du[2] = - x_c * C * ( 1. - y_c * R / (R + R_0)) - x_p * y_p * P * C / (C + C_0) #C
    du[3] = - x_p * P + x_p * y_p * C * P / (C + C_0) #P
end
