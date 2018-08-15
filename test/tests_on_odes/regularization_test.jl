using PenaltyFunctions, Optim, LinearAlgebra

cost_function_1 = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                   Regularization(0.6,L2Penalty()),maxiters=10000,verbose=false)
cost_function_2 = build_loss_objective(prob2,Tsit5(),L2Loss(t,data),verbose=false,
                   Regularization(0.1,MahalanobisPenalty(Matrix(1.0I, 2, 2))),maxiters=10000)
cost_function_3 = build_loss_objective(prob3,Tsit5(),L2Loss(t,data),verbose=false,
                   Regularization(0.1,MahalanobisPenalty(Matrix(1.0I, 4, 4))),maxiters=10000)

println("Use Optim BFGS to fit the parameter")
result = Optim.optimize(cost_function_1, [1.0], Optim.BFGS())
@test result.minimizer[1] ≈ 1.5 atol=3e-1

result_bfgs = Optim.optimize(cost_function_2, [1.2,2.7], Optim.BFGS())
@test result_bfgs.minimizer ≈ [1.5;3.0] atol=3e-1

result_bfgs = Optim.optimize(cost_function_3, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result_bfgs.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1
