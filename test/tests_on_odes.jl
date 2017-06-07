### Optim Method
import Optim

println("Use Optim Brent to fit the parameter")
result = Optim.optimize(obj, 1.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Use Optim BFGS to fit the parameter")
result = Optim.optimize(obj, [1.0], Optim.BFGS())
@test result.minimizer[1] ≈ 1.5 atol=3e-1
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)

import LeastSquaresOptim
println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
x = [1.0]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(x = x,
                f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Use Two Stage Method to fit the parameter")
cost_function = two_stage_method(prob,t,data)
result = Optim.optimize(cost_function, 0.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1


println("Use Optim BFGS to fit the parameter")
cost_function = build_loss_objective(prob,Tsit5(),CostVData(t,data),maxiters=10000)
result = Optim.optimize(cost_function, [1.0,2.5], Optim.BFGS())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1

println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
x = [1.3,2.7]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(x = x,
                f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test res.minimizer ≈ [1.5;3.0] atol=3e-1

println("Use Two Stage Method to fit the parameter")
cost_function = two_stage_method(prob,t,data)
result = Optim.optimize(cost_function, [1.0,2.5], Optim.BFGS())
@test result.minimizer ≈ [1.5;3.0] atol=3e-1

println("Multivariate")
f2 = @ode_def_nohes LotkaVolterraAll begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1.0 c=>3.0 d=>1.0

u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f2,u0,tspan)



println("Use Optim BFGS to fit the parameter")
srand(200)
cost_function = build_loss_objective(prob,Tsit5(),CostVData(t,data),maxiters=10000)
result = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1

println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
x = [1.3,0.8,2.8,1.2]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(
                x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test res.minimizer ≈ [1.5;1.0;3.0;1.0] atol=3e-1

println("Use Two Stage Method to fit the parameter")
cost_function = two_stage_method(prob,t,data)
result = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1
