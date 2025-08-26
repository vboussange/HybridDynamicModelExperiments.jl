#=
Running inference simulations with Model3SP, for different meta parameter values.

Inference simulations are ran in a distributed fashion; the first argument to
the script corresponds to the number of processes used.

```
julia scripts/inference_3sp/inference_3sp.jl 10
```
will run the script over 10 processes.
=#
cd(@__DIR__)
using Lux
using HybridModelling
using HybridModellingBenchmark; HybridModellingBenchmark.setup_distributed_environment()
import HybridModellingBenchmark: Model3SP, LuxBackend, MCMCBackend, InferICs, run_simulations
import OrdinaryDiffEq: Tsit5
import SciMLSensitivity: ForwardDiffSensitivity, GaussAdjoint
import ADTypes: AutoForwardDiff
import Optimisers: Adam
using Turing
using ProgressMeter
using Distributed
using Dates
using Random
using JLD2
using DataFrames
using Logging
Logging.disable_logging(Logging.Warn)

iterations = 5000
const TrueParameters = (H = Float32[1.24, 2.5],
                        q = Float32[4.98, 0.8],
                        r = Float32[1.0, -0.4, -0.08],
                        A = Float32[1.0])
const TrueInitialState = Float32[0.77, 0.060, 0.945]
const TimeSteps = range(500f0, step=4, length=100)
const TimeSpan = (0f0, TimeSteps[end])
fixed_params = (alg = Tsit5(),
                adtype = AutoForwardDiff(),
                abstol = 1e-4,
                reltol = 1e-4,
                tsteps = TimeSteps,
                verbose = false,
                maxiters = 50_000,
                sensealg = GaussAdjoint(), 
                p_true = TrueParameters, 
                rng = Random.MersenneTwister(1234),
                batchsize = 10)

function generate_data(;alg, abstol, reltol, tspan, tsteps, p_true, rng, kwargs...)
    parameters = ParameterLayer(constraint = NoConstraint(), 
                                init_value = p_true)
    lux_true_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((;u0 = TrueInitialState), ps, st)
    return synthetic_data
end

function create_simulation_parameters(iterations)
    segmentsizes = floor.(Int, exp.(range(log(2), log(100), length=6)))
    noises = 0:0.1:0.5
    nruns = 10
    batchsizes = [10]
    optim_backends = [LuxBackend(), MCMCBackend()]
    models = [Model3SP()]
    ic_estims = [InferICs(true), InferICs(false)]

    pars_arr = []
    for segmentsize in segmentsizes, noise in noises, run in 1:nruns, batchsize in batchsizes, optim_backend in optim_backends, infer_ic in ic_estims, model in models
        varying_params = (;)
        if isa(optim_backend, LuxBackend) 
            varying_params = merge(varying_params, (opt = Adam(5e-3), n_epochs = iterations)) 
        else
            varying_params = merge(varying_params, (sampler = HMC(0.05, 4; adtype=fixed_params.adtype), n_iterations = iterations))
        end
        varying_params = merge(varying_params, (;segmentsize, 
                                                noise, 
                                                batchsize, 
                                                optim_backend, 
                                                experimental_setup = infer_ic, 
                                                model))
        push!(pars_arr, varying_params)
    end
    return pars_arr
end

data = generate_data(;tspan = TimeSpan, fixed_params...)
# using Plots; plot(fixed_params.tsteps, data')
fixed_params = merge(fixed_params, (;data) )

println("Warming up...")
simulation_parameters_warmup = create_simulation_parameters(1)
run_simulations(simulation_parameters_warmup; fixed_params...) # precompilation for std model


println("Starting simulations...")
simulation_parameters = create_simulation_parameters(5000)
results = run_simulations(simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results, Epochs, model_true, data)
