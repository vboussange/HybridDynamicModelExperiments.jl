include("composable_ecosystem_model.jl")
using Test
tspan = (0.,1000.)
# /!\ parameters must be represented as vector
f1 = f_I(); f2 = f_II(); f3 = f_BedAng(); f4 = NullResp()

x = 1.; y = 2; p = ones(3)
@test all([f(x,y,p) isa Number for f in [f1, f2, f3, f4]])
@test f4(1., []) + f1(1., [2.]) == f1(1., [2.])
@test f3(x,y,p) == 0.25


# Testing only predatory
p1 = [0.1, 0.2]
p2 = [0.1]
ds = [0.2, 0.015]
u0 = [0.8,0.2,8.]
em = EcosystemModel(f1, f2, p1, p2, ds)
@test length(Flux.params(em)...) == 5
prob = ODEProblem(em, u0, tspan, em.p)
@time sol = solve(prob, alg=Tsit5(), sensealg=ForwardDiffSensitivity())
using Plots
Plots.plot(sol)



# Testing only omnivory
p1 = [0.1, 0.2]
p2 = [0.1]
p3 = [0.1, 0.2, 0.3]
ds = [0.2, 0.015]
u0 = [0.8,0.2,8.]
em = EcosystemModel(f1, f2, f3, p1, p2, p3, ds)
@test length(Flux.params(em)...) == 8
prob = ODEProblem(em, u0, tspan, em.p)
@time sol = solve(prob, alg=Tsit5(), sensealg=ForwardDiffSensitivity())
using Plots
Plots.plot(sol)


# ========================================
# Testing EcosystemModelOmnivory
alg = Tsit5()
abstol = 1e-6
reltol = 1e-6
tspan = (0.,1000.)
u0 = [0.8,0.2,0.8]
##########################
##### system of hasting ##
##########################
em = EcosystemModelOmnivory(x_c = 0.4, 
                            x_p = 0.08, 
                            y_c = 2.009, 
                            y_pr = 2.00, 
                            y_pc = 5., 
                            R_0 = 0.16129, 
                            R_02 = 0.5, 
                            C_0 = 0.5, 
                            ω = 0.1)
prob = ODEProblem(em, u0, tspan, em.p)
@time em_true_sol = solve(prob, alg = alg,  abstol=abstol, reltol=reltol)
using Plots
Plots.plot(em_true_sol)

##########################
##### system of McKann ##
##########################
# plausible x_p : 0.071 to 0.225
# other R_0 and C_0 can be decreased to find chaos
em = EcosystemModelMcKann(x_c = 0.4, 
                            x_p = 0.072, 
                            y_c = 2.01, 
                            y_p = 5.00, 
                            R_0 = 0.16129, 
                            C_0 = 0.5)
prob = ODEProblem(em, u0, tspan, em.p)
@time em_true_sol = solve(prob, alg = alg,  abstol=abstol, reltol=reltol)
using Plots
Plots.plot(em_true_sol)


# params of hasting 1991 (non plausible)
u0 = [0.8,0.2,8.]
em = EcosystemModelMcKann(x_c = 0.4, 
                            x_p = 0.01, 
                            y_c = 2.01, 
                            y_p = 5.00, 
                            R_0 = 0.16129, 
                            C_0 = 0.5)
prob = ODEProblem(em, u0, tspan, em.p)
@time em_true_sol = solve(prob, alg = alg,  abstol=abstol, reltol=reltol)
using Plots
Plots.plot(em_true_sol)