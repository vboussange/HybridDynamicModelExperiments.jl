using Lux
using HybridModellingExperiments
using HybridModelling
import HybridModellingExperiments: Model3SP, LuxBackend, MCMCBackend, InferICs, run_simulations, LogMSELoss, save_results
import HybridModellingExperiments: SerialMode, ParallelMode
import OrdinaryDiffEq: Tsit5
import SciMLSensitivity: ForwardDiffSensitivity
import ADTypes: AutoForwardDiff
import Optimisers: Adam

using ProgressMeter
using Dates
using Random
using JLD2
using DataFrames

const p_true = (H = [1.24, 2.5],
                        q = [4.98, 0.8],
                        r = [1.0, -0.4, -0.08],
                        A = [1.0])
const u0_true = [0.77, 0.060, 0.945]
const tsteps = range(500e0, step=4, length=111)
const tspan = (0e0, tsteps[end])

fixed_params = (alg = Tsit5(),
                adtype = AutoForwardDiff(),
                abstol = 1e-4,
                reltol = 1e-4,
                tsteps,
                verbose = false,
                maxiters = 50_000,
                sensealg = ForwardDiffSensitivity(), 
                p_true, 
                rng = Random.MersenneTwister(1234),
                batchsize = 10,
                optim_backend = LuxBackend(),
                n_epochs = 5000,
                verbose_frequency = Inf,
                loss_fn = LogMSELoss(),
                forecast_length = 10)

function generate_data(;alg, abstol, reltol, tspan, tsteps, p_true, rng, kwargs...)
    parameters = ParameterLayer(constraint = NoConstraint(), 
                                init_value = p_true)
    lux_true_model = ODEModel((;parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((;u0 = u0_true), ps, st)
    return synthetic_data
end

function create_simulation_parameters()
    segmentsizes = floor.(Int, exp.(range(log(2), log(100), length=6)))
    nruns = 10
    models = [Model3SP()]
    ic_estims = [InferICs(true), InferICs(false)]
    lrs = [1e-3, 1e-2, 1e-1]
    noises = [0.3]

    pars_arr = []
    for segmentsize in segmentsizes, run in 1:nruns, infer_ic in ic_estims, model in models, lr in lrs, noise in noises
        varying_params = (;segmentsize,
                            experimental_setup = infer_ic, 
                            opt = Adam(lr), 
                            model,
                            noise)
        push!(pars_arr, varying_params)
    end
    return shuffle!(fixed_params.rng, pars_arr)
end

data = generate_data(;tspan, fixed_params...)
# using Plots; plot(fixed_params.tsteps, data')
fixed_params = merge(fixed_params, (;data))

# println("Warming up...")
# simulation_parameters_warmup = create_simulation_parameters(1)
# run_simulations(simulation_parameters_warmup; fixed_params...) # precompilation for std model


println("Starting simulations...")
simulation_parameters = create_simulation_parameters()
results = run_simulations(ParallelMode(), simulation_parameters; fixed_params...)

save_results(string(@__FILE__); results)
