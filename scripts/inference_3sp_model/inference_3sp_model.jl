#=
Running inference simulations with SimpleEcosystemModel3SP, for different meta parameter values.

Inference simulations are ran in a distributed fashion; the first argument to
the script corresponds to the number of processes used.

```
julia scripts/inference_3sp/inference_3sp.jl 10
```
will run the script over 10 processes.
=#
cd(@__DIR__)
using JLD2, DataFrames, Random, Dates, ProgressMeter, Distributed, ComponentArrays
using Logging
Logging.disable_logging(Logging.Warn)

include("../../src/utils.jl")
include("../../src/run_simulations.jl")

const SimulFile = "simul.jl"
const Epochs = [5000]
const TrueParameters = ComponentArray(H = Float32[1.24, 2.5],
                                      q = Float32[4.98, 0.8],
                                      r = Float32[1.0, -0.4, -0.08],
                                      A = Float32[1.0])
const TrueInitialState = Float32[0.77, 0.060, 0.945]
const TimeSteps = range(500f0, step=4, length=100)
const TimeSpan = (0f0, TimeSteps[end])
const MyModel = SimpleEcosystemModel3SP

Random.seed!(5)


function generate_model_params()
    return (alg = Tsit5(),
            abstol = 1e-4,
            reltol = 1e-4,
            tspan = TimeSpan,
            saveat = TimeSteps,
            verbose = false,
            maxiters = 50_000)
end

function generate_inference_params()
    return (tsteps = TimeSteps,
            optimizers = [ADAM(5e-3)],
            verbose_loss = true,
            info_per_its = 1000,
            multi_threading = false)
end

function generate_df_results()
    DataFrame(group_size = Int[], 
            batchsize = Int[],
            noise = Float64[], 
            med_par_err = Float64[], 
            loss = Float64[], 
            time = Float64[], 
            res = Any[], 
            adtype = Any[])
end

function create_simulation_parameters()
    group_sizes = floor.(Int,exp.(range(log(2), log(100), length=6)))
    noises = 0:0.1:0.5
    nruns = 10
    adtypes = [Optimization.AutoZygote()]
    batchsizes = [10]

    pars_arr = Dict{String,Any}[]
    for group_size in group_sizes, noise in noises, run in 1:nruns, adtype in adtypes, batchsize in batchsizes
        sensealg = typeof(adtype) <: AutoZygote ? BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)) : nothing
        model_params = generate_model_params()
        model = MyModel(ModelParams(; p=TrueParameters, u0=TrueInitialState, sensealg, model_params...))
        push!(pars_arr, Dict("group_size" => group_size, "batchsize" => batchsize, "noise" => noise, "adtype" => adtype, "model" => model))
    end
    return shuffle(pars_arr)
end

setup_distributed_environment(SimulFile)
simulation_parameters = create_simulation_parameters()
model_true = MyModel(ModelParams(; p=TrueParameters, tspan=TimeSpan, u0=TrueInitialState, saveat=TimeSteps, generate_model_params()...))

data = simulate(model_true, u0=TrueInitialState) |> Array
println("Warming up...")
run_simulations([simulation_parameters[1] for p in workers()], 1, data) # precompilation for std model

println("Starting simulations...")
results = run_simulations(simulation_parameters, Epochs, data)

save_results(string(@__FILE__); results, Epochs, model_true, data)
