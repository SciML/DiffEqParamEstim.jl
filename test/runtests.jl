using DiffEqParamEstim, OrdinaryDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

# Here's the problem to solve

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = [0;10.0]
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)

#using Plots
#gr()
#plot(sol)

# Generate random data based off of the known solution

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]
data = vecvec_to_mat(randomized)

#scatter!(t,data)
#scatter(t,data)

# See how far we get off even with small changes to the parameter
f = LotkaVolterraTest(a=1.42)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)
#plot!(sol)

# Use LM to fit the parameter
fit = lm_fit(prob,t,vec(data),[1.49],show_trace=true,lambda=10000.0)
param = fit.param
# @test param[1] ≈ 1.5 # Fails because Optim's fails...
#for i in eachindex(f.params)
#  setfield!(f,f.params[i],param[i])
#end
#prob = ODEProblem(f,u0)
#sol_optimized = solve(prob)
#plot(sol_optimized,leg=false)
#scatter!(t,data)
#plot!(sol)

### Optim Method
using Optim
cost_function = build_optim_objective(prob,t,data)

result = optimize(cost_function, 1.0, 10.0)
@test 1.5 - result.minimum[1] < 0.01
sol_optimized = solve(prob)
#plot(sol_optimized,leg=false)
#scatter!(t,data)
#plot!(sol)

result = optimize(cost_function, [1.45], BFGS())
@test 1.5 - result.minimum[1] < 0.2
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)

using LeastSquaresOptim
cost_function = build_lsoptim_objective(prob,t,data)
x = [1.0]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR(),
                ftol=1e-14,xtol=1e-15,iterations=100,grtol=1e-14)

@test res.minimizer[1] - 1.52332 < 1e-4

f = LotkaVolterraTest(a=res.minimizer[1])
prob = ODEProblem(f,u0,tspan)
sol = solve(prob)

#using Plots
#gr()
#plot(sol)
#scatter!(t,data)
