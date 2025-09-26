using Lux
using HybridDynamicModelExperiments
using HybridDynamicModels
import HybridDynamicModelExperiments: Model3SP, SGDBackend, MCSamplingBackend, InferICs, run_simulations, LogMSELoss, save_results
import HybridDynamicModelExperiments: SerialMode, ParallelMode
import OrdinaryDiffEqTsit5: Tsit5
import SciMLSensitivity: ForwardDiffSensitivity, BacksolveAdjoint, ReverseDiffVJP
import ADTypes: AutoForwardDiff
import Optimisers: Adam
import Turing: HMC
import Distributions: LogNormal

using ProgressMeter
using Dates
using Random
using JLD2
using DataFrames

mode = SerialMode()
const p_true = (H = [1.24, 2.5],
                    q = [4.98, 0.8],
                    r = [1.0, -0.4, -0.08],
                    A = [1.0])
const u0_true = [0.77, 0.060, 0.945]
const tsteps = range(500e0, step=4, length=111)
const tspan = (0e0, tsteps[end])
const n_iterations = 2000
datadistrib = x -> LogNormal(log(max(x, 1e-6)))

fixed_params = (alg = Tsit5(),
                abstol = 1e-4,
                reltol = 1e-4,
                tsteps,
                maxiters = 50_000,
                sensealg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true)), 
                p_true, 
                rng = Random.MersenneTwister(1234),
                optim_backend = MCSamplingBackend(HMC(0.05, 4; adtype=AutoForwardDiff()), 
                                            n_iterations, 
                                            datadistrib; progress = false),
                loss_fn = LogMSELoss(),
                forecast_length = 10,
                noise = 0.2,
                experimental_setup = InferICs(false))

function generate_data(;alg, abstol, reltol, tspan, tsteps, p_true, rng, kwargs...)
    parameters = ParameterLayer(init_value = p_true)
    lux_true_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((;u0 = u0_true), ps, st)
    return synthetic_data
end

function create_simulation_parameters()
    segment_lengths = floor.(Int, exp.(range(log(2), log(100), length=6)))
    models = [Model3SP()]
    nruns = 5

    pars_arr = []
    for segment_length in segment_lengths, model in models, _ in 1:nruns
        varying_params = (;segment_length,
                            model)
        push!(pars_arr, varying_params)
    end
    return shuffle!(fixed_params.rng, pars_arr)
end

data = generate_data(;tspan, fixed_params...)
fixed_params = merge(fixed_params, (;data))

simulation_parameters = create_simulation_parameters()
println("Created $(length(simulation_parameters)) simulations...")
println("Starting simulations...")
results = run_simulations(mode, simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)
