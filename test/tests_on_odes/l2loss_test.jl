using BlackBoxOptim, Optim

cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
bound1 = Tuple{Float64, Float64}[(1, 2)]
result = bboptimize(cost_function;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate[1] ≈ 1.5 atol=3e-1

cost_function = build_loss_objective(prob2,Tsit5(),L2Loss(t,data,differ_weight=nothing,data_weight=1.0),
                                     maxiters=10000,verbose=false)
bound2 = Tuple{Float64, Float64}[(1, 2),(1, 4)]
result = bboptimize(cost_function;SearchRange = bound2, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5;3.0] atol=3e-1

cost_function = build_loss_objective(prob3,Tsit5(),L2Loss(t,data,differ_weight=10),
                                     maxiters=10000,verbose=false)
bound3 = Tuple{Float64, Float64}[(1, 2),(0, 2), (2, 4), (0, 2)]
result = bboptimize(cost_function;SearchRange = bound3, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5;1.0;3.0;1.0] atol=5e-1

cost_function = build_loss_objective(prob3,Tsit5(),L2Loss(t,data,differ_weight=0.3,data_weight=0.7),
                                     maxiters=10000,verbose=false)
bound3 = Tuple{Float64, Float64}[(1, 2),(0, 2), (1, 4), (0, 2)]
result = bboptimize(cost_function;SearchRange = bound3, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5;1.0;3.0;1.0] atol=5e-1

println("LSA for gradient")
cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false,flsa_gradient=true)
bound1 = Tuple{Float64, Float64}[(1, 2)]
function g_!(stor,x)
    cost_function.cost_function2(x,stor)
end
result = Optim.optimize(cost_function.cost_function,g_!,[1.0], Optim.ConjugateGradient())
@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Adjoint SA for gradient")
cost_function = build_loss_objective(prob2,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false,adjsa_gradient=true)
function g_!(stor,x)
    cost_function.cost_function2(x,stor)
end
result = Optim.optimize(cost_function.cost_function,g_!,[1.3,2.0], Optim.ConjugateGradient())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1