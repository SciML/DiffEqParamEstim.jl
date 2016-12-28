using DiffEqParamEstim, OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

# Here's the problem to solve

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob,Tsit5())

# Generate random data based off of the known solution

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]
data = vecvec_to_mat(randomized)

println("Use LM to fit the parameter")
fit = lm_fit(prob,t,vec(data),[1.0],Tsit5(),show_trace=true,lambda=10000.0)
param = fit.param
@test_approx_eq_eps param[1] 1.5 1e-3

### Optim Method
using Optim
cost_function = build_optim_objective(prob,t,data,Tsit5(),maxiters=10000)

println("Use Optim Brent to fit the parameter")
result = optimize(cost_function, 1.0, 10.0)
@test_approx_eq_eps result.minimizer[1] 1.5 3e-1

println("Use Optim BFGS to fit the parameter")
result = optimize(cost_function, [1.0], BFGS())
@test_approx_eq_eps result.minimizer[1] 1.5 3e-1
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)

using LeastSquaresOptim
println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
x = [1.0]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test_approx_eq_eps result.minimizer[1] 1.5 3e-1


println("Multivariate")

f2 = @ode_def_nohes LotkaVolterraMulti begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=>3.0 d=1.0

u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f2,u0,tspan)

println("Use LM to fit the parameter")
fit = lm_fit(prob,t,vec(data),[1.3,2.6],Tsit5(),show_trace=true,lambda=10000.0)
param = fit.param
@test_approx_eq_eps param [1.5;3.0] 1e-3

println("Use Optim BFGS to fit the parameter")
cost_function = build_optim_objective(prob,t,data,Tsit5(),maxiters=10000)
result = optimize(cost_function, [1.0,2.5], BFGS())
@test_approx_eq_eps result.minimizer [1.5;3.0] 3e-1

println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob,t,data,Tsit5())
x = [1.3,2.7]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test_approx_eq_eps res.minimizer [1.5;3.0] 3e-1
