include("test_problems.jl")

println("Use Two Stage Method to fit the parameter")
cost_function = two_stage_method(prob1,t,data)
result = Optim.optimize(cost_function, 0.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Multivariate")
cost_function = two_stage_method(prob2,t,data)
result = Optim.optimize(cost_function, [1.0,2.5], Optim.BFGS())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1

cost_function = two_stage_method(prob3,t,data)
result = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1
