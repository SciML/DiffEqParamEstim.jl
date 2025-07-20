using PenaltyFunctions, OptimizationOptimJL, LinearAlgebra, SciMLSensitivity

cost_function_1 = build_loss_objective(prob1, Tsit5(), L2Loss(t, data),
    Optimization.AutoZygote(),
    Regularization(0.6, L2Penalty()), maxiters = 10000,
    verbose = false, abstol = 1e-8, reltol = 1e-8)
cost_function_2 = build_loss_objective(prob2, Tsit5(), L2Loss(t, data),
    Optimization.AutoZygote(),
    Regularization(0.1,
        MahalanobisPenalty(Matrix(1.0I, 2, 2))),
    verbose = false,
    abstol = 1e-8, reltol = 1e-8,
    maxiters = 10000)
cost_function_3 = build_loss_objective(prob3, Tsit5(), L2Loss(t, data),
    Optimization.AutoZygote(),
    Regularization(0.1,
        MahalanobisPenalty(Matrix(1.0I, 4, 4))),
    verbose = false,
    abstol = 1e-8, reltol = 1e-8,
    maxiters = 10000)

println("Use Optim BFGS to fit the parameter")
optprob = Optimization.OptimizationProblem(cost_function_1, [1.0], lb=[1.0], ub=[2.0])
result = solve(optprob, Optim.BFGS())
@test result.u[1]≈1.5 atol=3e-1

optprob = Optimization.OptimizationProblem(cost_function_2, [1.2, 2.7], lb=[1.0, 2.0], ub=[2.0, 4.0])
result = solve(optprob, Optim.BFGS())
@test result.minimizer≈[1.5; 3.0] atol=3e-1

optprob = Optimization.OptimizationProblem(cost_function_3, [1.3, 0.8, 2.8, 1.2], lb=[1.0, 0.5, 2.0, 0.5], ub=[2.0, 1.5, 4.0, 1.5])
result = solve(optprob, Optim.BFGS())
@test result.minimizer≈[1.5; 1.0; 3.0; 1.0] atol=5e-1
