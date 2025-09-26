module HybridDynamicModelExperiments
using Lux, Optimisers
using HybridDynamicModels
using Random

include("generics.jl")
include("utils.jl")
include("lux_trainer.jl")
include("loss_fn.jl")

include("models/ecosystem_model.jl")
include("models/3sp_model.jl")
include("models/5sp_model.jl")
include("models/7sp_model.jl")
include("models/hybrid_functional_response_model.jl")
include("models/hybrid_growth_rate_model.jl")

include("run_simulations.jl")
include("simul.jl")

export Model3SP, Model5SP, Model7SP, HybridFunctionalResponseModel, HybridGrowthRateModel
export simu
export MCSamplingBackend, SGDBackend, train, sample

end
