using OrdinaryDiffEq, Test, SciMLSensitivity, Optimization, OptimizationOptimJL,
    DiffEqParamEstim
using Random: Xoshiro
using SciMLBase: successful_retcode

function LotkaVolterraTest_not_inplace(u, a, t)
    b, c, d = 1.0, 3.0, 1.0
    x, y = u[1], u[2]
    du = zeros(eltype(u), 2)
    du[1] = a[1] * x - b * x * y
    du[2] = -c * y + d * x * y
    return du
end

# forward
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
true_parameter = 1.5
prob = ODEProblem(LotkaVolterraTest_not_inplace, u0, tspan, true_parameter)

solver_tolerances = (; abstol = 1.0e-8, reltol = 1.0e-8)
sol = solve(prob, Tsit5(); solver_tolerances...)
t = collect(range(0, stop = 10, length = 200))
using RecursiveArrayTools # for VectorOfArray
rng = Xoshiro(18)
randomized = VectorOfArray([sol(ti) + 0.01randn(rng, 2) for ti in t])
data = convert(Array, randomized)

# inverse
cost_function = build_loss_objective(
    prob, Tsit5(), L2Loss(t, data),
    Optimization.AutoZygote();
    maxiters = 10000, solver_tolerances...
)
optprob = Optimization.OptimizationProblem(cost_function, [1.0], lb = [0.0], ub = [10.0])
fit = solve(optprob, BFGS(); maxiters = 15, local_maxiters = 30)
@test successful_retcode(fit)
@test only(fit.u) ≈ true_parameter atol = 0.001
@test fit.stats.fevals <= 250

# two-stage OOP regression test

function ff(du, u, p, t)
    return du .= p .* u
end
function ff(u, p, t)
    return p .* u
end
rc = 62
ps = repeat([-0.001], rc)
tspan = (7.0, 84.0)
u0 = 3.4 .+ ones(rc)
t = collect(range(minimum(tspan), stop = maximum(tspan), length = 157))
prob = ODEProblem(ff, u0, tspan, ps)
prob_oop = ODEProblem{false}(ff, u0, tspan, ps)
data = Array(solve(prob, Tsit5(), saveat = t))
ptest = ones(rc)

obj_ts = two_stage_objective(prob, t, data; kernel = :Sigmoid)
@test obj_ts(ptest) ≈ 418.3400017500223^2
obj_ts = two_stage_objective(prob_oop, t, data; kernel = :Sigmoid)
@test obj_ts(ptest) ≈ 418.3400017500223^2
