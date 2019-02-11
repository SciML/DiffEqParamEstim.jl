using NLopt

println("Use NLOpt to fit the parameter")

obj = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                           maxiters=10000,verbose=false)

opt = Opt(:LN_COBYLA, 1)
min_objective!(opt, obj)
(minf,minx,ret) = NLopt.optimize(opt,[1.4])
@test minx[1] ≈ 1.5 atol=1e-3

opt = Opt(:GN_ESCH, 1)
min_objective!(opt, obj.cost_function2)
lower_bounds!(opt,[1.0])
upper_bounds!(opt,[3.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 10000)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
@test minx[1] ≈ 1.5 atol=1e-1

opt = Opt(:GN_ISRES, 1)
min_objective!(opt, obj.cost_function2)
lower_bounds!(opt,[1.0])
upper_bounds!(opt,[3.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 100000)
(minf,minx,ret) = NLopt.optimize(opt,[1.2])
@test minx[1] ≈ 1.5 atol=1e-1

# test differentiation

for autodiff in (false, true)
   global obj = build_loss_objective(prob1, Tsit5(), L2Loss(t,data);
                              mpg_autodiff = autodiff, maxiters = 10000)

   global opt = Opt(:LD_MMA, 1)
   min_objective!(opt, obj.cost_function2)
   xtol_rel!(opt,1e-3)
   maxeval!(opt, 10000)
   global minf,minx,ret
   (minf,minx,ret) = NLopt.optimize(opt, [1.3])
   @test minx[1] ≈ 1.5 atol=5e-1 #take a look at this, it behaves weirdly
end
