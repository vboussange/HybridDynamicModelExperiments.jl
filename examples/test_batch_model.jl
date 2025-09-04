using HybridModellingExperiments
using HybridModelling
using LuxCUDA, Lux
using SciMLSensitivity
import OrdinaryDiffEqTsit5: Tsit5
using Random
# Problem: we use scalar indexing within model definition, which is not allowed on GPU
LuxCUDA.CUDA.allowscalar(true)

p_true = (H = [1.24, 2.5],
    q = [4.98, 0.8],
    r = [1.0, -0.4, -0.08],
    A = [1.0])
u0_true = [0.77, 0.060, 0.945]
tsteps = range(500e0, step = 4, length = 111)
tspan = (0e0, tsteps[end])

device = gpu_device()

alg = Tsit5()
adtype = AutoZygote()
abstol = 1e-4
reltol = 1e-4
tsteps
verbose = false
maxiters = 50_000
sensealg = BacksolveAdjoint(autojacvec = ZygoteVJP(true))
rng = Random.MersenneTwister(1234)


parameters = ParameterLayer(init_value = p_true)
lux_true_model = ODEModel(
    (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

ps, st = Lux.setup(rng, lux_true_model)
ps, st = device(ps), device(st)
u0_true = device(u0_true)
synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)