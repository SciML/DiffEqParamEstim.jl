using NLopt

println("Use Multiple Shooting Method the parameter")

multiple_shooting_objective_constraints = multiple_shooting_method(prob1,Tsit5(),L2Loss(t,data),maxiters=10000)
opt = Opt(:LN_COBYLA, 1)
min_objective!(opt, multiple_shooting_objective_constraints.multiple_shooting_cost)
timestamp = collect(linespace(prob1.tspan[1],prob1.tspan[2],10)
for i in 1:(length(timestamp)-1)*length(prob1.u0)
  equality_constraint!(opt, multiple_shooting_objective_constraints.constraints[i], 1e-8)
end

(minf,minx,ret) = NLopt.optimize(opt,[1.3])
#@test minx[1] ≈ 1.5 atol=1e-3
