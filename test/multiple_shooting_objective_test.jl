using OrdinaryDiffEq, DiffEqParamEstim, Distributions, Zygote,
    Optimization, OptimizationBBO, OptimizationOptimJL
ms_f = function (du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    return du[2] = -3.0 * u[2] + u[1] * u[2]
end
ms_u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
ms_p = [1.5, 1.0]
ms_prob = ODEProblem(ms_f, ms_u0, tspan, ms_p)
t = collect(range(0, stop = 10, length = 200))
data = Array(solve(ms_prob, Tsit5(), saveat = t, abstol = 1.0e-12, reltol = 1.0e-12))
bound = Tuple{Float64, Float64}[
    (0, 10), (0, 10), (
        0, 10,
    ), (0, 10),
    (0, 10), (0, 10), (0, 10), (
        0, 10,
    ),
    (0, 10), (0, 10), (0, 10), (0, 10),
    (
        0, 10,
    ), (0, 10), (0, 10), (0, 10), (0, 10), (
        0, 10,
    ),
]

ms_obj = multiple_shooting_objective(
    ms_prob, Tsit5(), L2Loss(t, data),
    Optimization.AutoZygote();
    discontinuity_weight = 1.0, abstol = 1.0e-12,
    reltol = 1.0e-12
)
optprob = Optimization.OptimizationProblem(
    ms_obj, fill(5.0, 18),
    lb = first.(bound), ub = last.(bound)
)
result = solve(
    optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxiters = 7000
)
@test result.u[(end - 1):end] ≈ [1.5, 1.0] atol = 2.0e-1

priors = [Truncated(Normal(1.5, 0.5), 0, 2), Truncated(Normal(1.0, 0.5), 0, 1.5)]
ms_obj1 = multiple_shooting_objective(
    ms_prob, Tsit5(), L2Loss(t, data),
    Optimization.AutoForwardDiff(); priors = priors,
    discontinuity_weight = 1.0, abstol = 1.0e-6,
    reltol = 1.0e-6
)
optprob = Optimization.OptimizationProblem(
    ms_obj1, zeros(18), lb = first.(bound),
    ub = last.(bound)
)
result = solve(optprob, BFGS(), maxiters = 500)
@test result.u[(end - 1):end] ≈ [1.5, 1.0] atol = 2.0e-1
