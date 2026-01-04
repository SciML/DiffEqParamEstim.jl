using DelayDiffEq, OrdinaryDiffEq, RecursiveArrayTools, Test,
    Sundials

function f(out, du, u, p, t)
    out[1] = -p[1] * u[1] + 1.0e4 * u[2] * u[3] - du[1]
    out[2] = +p[1] * u[1] - 3.0e7 * u[2]^2 - 1.0e4 * u[2] * u[3] - du[2]
    return out[3] = u[1] + u[2] + u[3] - 1.0
end
u₀ = [1.0, 0, 0]
du₀ = [-0.04, 0.04, 0.0]
tspan = (0.0, 100000.0)
differential_vars = [true, true, false]
prob = DAEProblem(f, du₀, u₀, tspan, [0.04], differential_vars = differential_vars)
sol = solve(prob, DFBDF())

t = collect(range(0, stop = 10, length = 30))
randomized = VectorOfArray([(sol(t[i]) + 0.003randn(3)) for i in 1:length(t)])
data = convert(Array, randomized)

using DiffEqParamEstim, OptimizationNLopt, OptimizationOptimJL, ForwardDiff, Zygote,
    Optimization, SciMLSensitivity
cost_function = build_loss_objective(
    prob, DFBDF(), L2Loss(t, data),
    Optimization.AutoZygote(), abstol = 1.0e-8,
    reltol = 1.0e-8, verbose = false
)
optprob = Optimization.OptimizationProblem(cost_function, [0.01]; lb = [0.0], ub = [1.0])
res = solve(optprob, OptimizationOptimJL.BFGS())

cost_function = build_loss_objective(
    prob, DFBDF(), L2Loss(t, data),
    Optimization.AutoForwardDiff(), abstol = 1.0e-8,
    reltol = 1.0e-8, verbose = false
)
optprob = Optimization.OptimizationProblem(cost_function, [0.01]; lb = [0.0], ub = [1.0])
res = solve(optprob, OptimizationOptimJL.BFGS())
@test res.u[1] ≈ 0.04 atol = 5.0e-3

# opt = Opt(:GN_ESCH, 1)
# res = solve(optprob, opt)
# @test res.u[1]≈0.04 atol=5e-3
