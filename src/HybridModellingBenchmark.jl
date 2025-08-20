module HybridModellingBenchmark
    __precompile__(false)

    using Lux
    using Random
    
    include("generics.jl")
    include("utils.jl")
    include("lux_trainer.jl")
    include("turing_trainer.jl")

    include("3sp_model.jl")
    include("5sp_model.jl")
    include("7sp_model.jl")
    include("hybrid_functional_response_model.jl")
    include("hybrid_growth_rate_model.jl")
    include("loss_fn.jl")

    include("plotting.jl")

    include("run_simulations.jl")
    include("simul.jl")

    export Model3SP, Model5SP, Model7SP, HybridFunctionalResponseModel, HybridGrowthRateModel
    export simu

end
