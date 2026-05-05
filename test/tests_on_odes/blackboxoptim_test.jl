using Optimization, OptimizationBBO

println("Use BlackBoxOptim to fit the parameter")
cost_function = build_loss_objective(
    prob1, Tsit5(), L2Loss(t, data),
    maxiters = 10000
)
bound1 = Tuple{Float64, Float64}[(1, 2)]
optprob = Optimization.OptimizationProblem(
    cost_function, [1.5], lb = first.(bound1), ub = last.(bound1)
)
result = solve(
    optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxiters = 5000
)
@test result.u[1] ≈ 1.5 atol = 3.0e-1

cost_function = build_loss_objective(
    prob2, Tsit5(), L2Loss(t, data),
    maxiters = 10000
)
bound2 = Tuple{Float64, Float64}[(1, 2), (2, 4)]
optprob = Optimization.OptimizationProblem(
    cost_function, [1.5, 3.0], lb = first.(bound2), ub = last.(bound2)
)
result = solve(
    optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxiters = 5000
)
@test result.u ≈ [1.5; 3.0] atol = 3.0e-1

cost_function = build_loss_objective(
    prob3, Tsit5(), L2Loss(t, data),
    maxiters = 10000
)
bound3 = Tuple{Float64, Float64}[
    (1, 2), (0, 2), (
        2, 4,
    ), (0, 2),
]
optprob = Optimization.OptimizationProblem(
    cost_function, [1.5, 1.0, 3.0, 1.0],
    lb = first.(bound3), ub = last.(bound3)
)
result = solve(
    optprob, BBO_adaptive_de_rand_1_bin_radiuslimited(),
    maxiters = 5000
)
@test result.u ≈ [1.5; 1.0; 3.0; 1.0] atol = 5.0e-1
