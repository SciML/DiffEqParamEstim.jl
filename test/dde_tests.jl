using DelayDiffEq, OrdinaryDiffEq, RecursiveArrayTools, Test

function f_lotka(du, u, h, p, t)
    du[1] = 0.5 * u[1] - p[1] * u[1] * u[2]
    return du[2] = p[2] * h(p, t - 0.5)[1] * u[2] - 0.5 * u[2]
end
h = (p, t) -> [0.5, 0.5]
u0 = [0.5, 0.5]
tspan = (0.0, 10.0)
p = [0.5, 1.0]

prob = DDEProblem(
    f_lotka, u0, h, tspan, p,
    constant_lags = [0.5]
)
sol = solve(prob, MethodOfSteps(Tsit5()))

t = collect(range(0, stop = 10, length = 30))
randomized = VectorOfArray([(sol(t[i]) + 0.03randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)

using DiffEqParamEstim, OptimizationNLopt

function f_lotka2(du, u, h, p, t)
    du[1] = 0.5 * u[1] - p[1] * u[1] * u[2]
    return du[2] = h(p, t - 0.5)[1] * u[2] - 0.5 * u[2]
end
h = (p, t) -> [0.5, 0.5]
u0 = [0.5, 0.5]
p = [0.5]

prob_opt = DDEProblem(f_lotka2, u0, h, tspan, p, constant_lags = [0.5])
cost_function = build_loss_objective(
    prob_opt, MethodOfSteps(Tsit5()),
    L2Loss(t, data), Optimization.AutoZygote(),
    abstol = 1.0e-8,
    reltol = 1.0e-8
)

optprob = Optimization.OptimizationProblem(cost_function, [1.0], lb = [0.0], ub = [1.0])
opt = Opt(:GN_ESCH, 1)
res = solve(optprob, BFGS())

@test res.u[1] ≈ 0.5 atol = 5.0e-3
