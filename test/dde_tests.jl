using DelayDiffEq, OrdinaryDiffEq, RecursiveArrayTools, Test

function f_lotka(du,u,h,p,t)
    du[1] = 0.5*u[1] - p[1]*u[1]*u[2]
    du[2] = p[2]*h(p,t-0.5)[1]*u[2] - 0.5*u[2]
end
h = (p,t) -> [0.5, 0.5]
u0 = [0.5, 0.5]
tspan = (0., 10.)
p = [0.5, 1.]

prob = DDEProblem(f_lotka,u0,h,tspan,p,
                   constant_lags = [0.5])
sol = solve(prob, MethodOfSteps(Tsit5()))

t = collect(range(0, stop=10, length=30))
randomized = VectorOfArray([(sol(t[i]) + .03randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

using DiffEqParamEstim, NLopt

function f_lotka2(du,u,h,p,t)
    du[1] = 0.5*u[1] - p[1]*u[1]*u[2]
    du[2] = h(p,t-0.5)[1]*u[2] - 0.5*u[2]
end
h = (p,t) -> [0.5, 0.5]
u0 = [0.5, 0.5]
p = [0.5]

prob_opt = DDEProblem(f_lotka2, u0, h, tspan, p, constant_lags = [0.5])
cost_function = build_loss_objective(prob_opt, MethodOfSteps(Tsit5()),
            L2Loss(t,data),maxiter=10000,abstol=1e-8,reltol=1e-8,verbose=false)

opt = Opt(:GN_ESCH, 1)
min_objective!(opt, cost_function.cost_function2)
lower_bounds!(opt, [0.0])
upper_bounds!(opt, [1.0])
maxeval!(opt, 10000)

(minf, minx, ret) = NLopt.optimize(opt,[0.2])
@test minx[1] ≈ 0.5 atol=5e-3
