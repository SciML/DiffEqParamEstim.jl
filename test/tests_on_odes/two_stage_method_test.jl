using Optim, NLopt

println("Use Two Stage Method to fit the parameter")

cost_function = two_stage_method(prob1,t,data)
result = Optim.optimize(cost_function, 0.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1
cost_function = two_stage_method(prob2,t,data)
result = Optim.optimize(cost_function, [1.0,2.5], Optim.BFGS())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1
cost_function = two_stage_method(prob3,t,data)
result = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1
 # test differentiation
obj = two_stage_method(prob2,t,data; mpg_autodiff = true)
function g_!(stor,x)
  obj.cost_function2(x,stor)
end
result = Optim.optimize(obj.cost_function,g_!,[1.3,0.8], Optim.ConjugateGradient())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1
for autodiff in (false, true)
  obj = two_stage_method(prob2,t,data; mpg_autodiff = autodiff)
  opt = Opt(:LD_LBFGS, 2)
  min_objective!(opt, obj.cost_function2)
  (minf,minx,ret) = NLopt.optimize(opt, [1.0,2.5])
  @test minx ≈ [1.5;3.0] atol=3e-1
  obj = two_stage_method(prob3,t,data; mpg_autodiff = autodiff)
  opt = Opt(:LD_LBFGS, 4)
  min_objective!(opt, obj.cost_function2)
  (minf,minx,ret) = NLopt.optimize(opt, [1.3,0.8,2.8,1.2])
  @test minx ≈ [1.5;1.0;3.0;1.0] atol=5e-1
end
