using BlackBoxOptim

println("Use BlackBoxOptim to fit the parameter")
cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
bound1 = Tuple{Float64, Float64}[(1, 2)]
result = bboptimize(cost_function;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate[1] ≈ 1.5 atol=3e-1

cost_function = build_loss_objective(prob2,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
bound2 = Tuple{Float64, Float64}[(1, 2),(2, 4)]
result = bboptimize(cost_function;SearchRange = bound2, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5;3.0] atol=3e-1

cost_function = build_loss_objective(prob3,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
bound3 = Tuple{Float64, Float64}[(1, 2),(0, 2), (2, 4), (0, 2)]
result = bboptimize(cost_function;SearchRange = bound3, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5;1.0;3.0;1.0] atol=5e-1
