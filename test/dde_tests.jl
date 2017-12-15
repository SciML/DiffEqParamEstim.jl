using DelayDiffEq, OrdinaryDiffEq, RecursiveArrayTools, Base.Test

function f_lotka(t, u, h, p, du)
    du[1] = 0.5*u[1] - p[1]*u[1]*u[2]
    du[2] = p[2]*h(t-0.5)[1]*u[2] - 0.5*u[2]
end
h = (t) -> [0.5, 0.5]
u0 = [0.5, 0.5]
tspan = (0., 10.)

prob = DDEProblem((t,u,h,du)->f_lotka(t,u,h,[0.5, 1.],du),h,u0,tspan,[0.5])
sol = solve(prob, MethodOfSteps(Tsit5()))

t = collect(linspace(0,10,30))
randomized = VectorOfArray([(sol(t[i]) + .03randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

using DiffEqParamEstim, NLopt

prob_opt = DDEProblem((t,u,h,p,du)->f_lotka(t,u,h,[p[1],1.],du), h, u0, tspan,[0.5])
cost_function = build_loss_objective(prob_opt, MethodOfSteps(Tsit5()),
            L2Loss(t,data),maxiter=10000,abstol=1e-8,reltol=1e-8,verbose=false)

opt = Opt(:GN_ESCH, 1)
min_objective!(opt, cost_function.cost_function2)
lower_bounds!(opt, [0.0])
upper_bounds!(opt, [1.0])
maxeval!(opt, 10000)

(minf, minx, ret) = NLopt.optimize(opt,[0.2])
@test minx[1] ≈ 0.5 atol=5e-3
