weight = 1.0e-6

cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data,colloc_grad=colloc_grad(t, data)),
                                     maxiters=10000,verbose=false)
result = Optim.optimize(cost_function, 1., 2.)
@test result.minimizer ≈ 1.5 atol=3e-1


cost_function = build_loss_objective(prob2,Tsit5(),L2Loss(t,data,
  differ_weight=weight,data_weight=weight,colloc_grad=colloc_grad(t, data)),
                                     maxiters=10000,verbose=false)
result = Optim.optimize(cost_function, [1.3,2.8], Optim.BFGS())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1


cost_function = build_loss_objective(prob3,Tsit5(),L2Loss(t,data,
  differ_weight=weight,colloc_grad=colloc_grad(t, data)), maxiters=10000,verbose=false)
result = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result.minimizer ≈ [1.5,1.0,3.0,1.0] atol=3e-1


cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data,
  data_weight=weight,colloc_grad=colloc_grad(t, data)),maxiters=10000,verbose=false)
result = Optim.optimize(cost_function, 1., 2)
@test result.minimizer ≈ 1.5 atol=3e-1
