using Optim
obj = build_loss_objective(prob1,Tsit5(),CostVData(t,data),maxiters=10000)

### Optim Method

println("Use Optim Brent to fit the parameter")
result = Optim.optimize(obj, 1.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Use Optim BFGS to fit the parameter")
result = Optim.optimize(obj, [1.0], Optim.BFGS())
@test result.minimizer[1] ≈ 1.5 atol=3e-1
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)

cost_function2 = build_loss_objective(prob2,Tsit5(),CostVData(t,data),maxiters=10000)
result_bfgs = Optim.optimize(cost_function2, [1.0,2.5], Optim.BFGS())
@test result_bfgs.minimizer ≈ [1.5;3.0] atol=3e-1

srand(200)
cost_function3 = build_loss_objective(prob3,Tsit5(),CostVData(t,data),maxiters=10000)
result_bfgs = Optim.optimize(cost_function3, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result_bfgs.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1

println("Use Optim NelderMead to fit the parameter")
result_neldermead = Optim.optimize(cost_function2, [1.0,2.5], Optim.NelderMead())
@test result_neldermead.minimizer ≈ [1.5;3.0] atol=3e-1



