import Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.develop(path="./ParametricModels.jl")
Pkg.develop(path="./PiecewiseInference.jl")