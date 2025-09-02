using Lux
using Turing
import Optimisers: Adam
import ADTypes: AutoZygote
import OrdinaryDiffEqTsit5: Tsit5
using SciMLSensitivity: BacksolveAdjoint, ReverseDiffVJP
using Random
using Test
import HybridModelling: ParameterLayer, ODEModel
import HybridModellingExperiments: Model3SP, LuxBackend, MCMCBackend, InferICs, LogMSELoss, init, train
import HybridModellingExperiments: simu, forecast

const p_true = (H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0])
const u0_true = [0.77, 0.060, 0.945]
const tsteps = range(500e0, step = 4, length = 111)
const tspan = (0e0, tsteps[end])

function generate_data(; alg, abstol, reltol, tspan, tsteps, p_true, rng, kwargs...)
    parameters = ParameterLayer(init_value = p_true)
    lux_true_model = ODEModel(
        (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

    ps, st = Lux.setup(rng, lux_true_model)
    synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)
    return synthetic_data
end


model = Model3SP()

myparams = (alg = Tsit5(),
            abstol = 1e-4,
            reltol = 1e-4,
            tsteps,
            verbose = false,
            maxiters = 50_000,
            sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)),
            p_true,
            rng = Random.MersenneTwister(1234),
            batchsize = 10,
            forecast_length = 10,
            perturb = 1e-4,
            segmentsize=3,
            model,
            noise = 0.)


data = generate_data(; tspan, myparams...)

optim_backend = LuxBackend(Adam(1e-5), 1, AutoZygote(), LogMSELoss())
infer_ics = InferICs(true)
dataloader = SegmentedTimeSeries((data, tsteps), segmentsize = myparams.segmentsize,
    batchsize = myparams.batchsize, partial_batch = true)
lux_model = init(model, optim_backend; p_true, myparams...).lux_model

res = train(optim_backend, lux_model, dataloader, infer_ics)
@test !isnothing(res) #TODO: improve test