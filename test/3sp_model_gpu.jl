using HybridModellingExperiments
using HybridModelling
using LuxCUDA, Lux

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
sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true))
p_true
rng = Random.MersenneTwister(1234)
batchsize = 10
n_epochs = 3000
loss_fn = LogMSELoss()
forecast_length = 10
perturb = 1e0

parameters = ParameterLayer(constraint = NoConstraint(),
    init_value = p_true)
lux_true_model = ODEModel(
    (; parameters), Model3SP(); alg, abstol, reltol, tspan, saveat = tsteps)

ps, st = Lux.setup(rng, lux_true_model)
ps, st = device(ps), device(st)
synthetic_data, _ = lux_true_model((; u0 = u0_true), ps, st)