cd(@__DIR__)
using JLD2
include("../load_data.jl")
include("../3sp_model.jl")
include("../5sp_model.jl")
include("../7sp_model.jl")

data, _, _, _, _ = load_data(SimpleEcosystemModel3SP)
@assert eltype(data) <: Float32

data, _, _, _, _ = load_data(SimpleEcosystemModel5SP)
@assert eltype(data) <: Float32

data, _, _, _, _ = load_data(SimpleEcosystemModel7SP)
@assert eltype(data) <: Float32
