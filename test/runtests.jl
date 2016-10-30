using ParameterEstimation, OrdinaryDiffEq, ParameterizedFunctions
using Base.Test

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = [0;10.0]

prob = ODEProblem(f,u0)
sol = solve(prob,tspan)

#using Plots
#gr()
#plot(sol)

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]

data = Matrix{Float64}(length(randomized),length(u0))
for i in 1:length(randomized)
  data[i,:] = randomized[i]
end
#scatter!(t,data)
#scatter(t,data)

f = LotkaVolterraTest(a=1.42)
prob = ODEProblem(f,u0)
sol = solve(prob,tspan)
#plot!(sol)

fit = lm_fit(prob,tspan,t,vec(data),[1.49],show_trace=true,lambda=10000.0)
param = fit.param
# @test param[1] ≈ 1.5 # Fails because Optim's fails...
#for i in eachindex(f.params)
#  setfield!(f,f.params[i],param[i])
#end
#prob = ODEProblem(f,u0)
#sol_optimized = solve(prob,tspan)
#plot(sol_optimized,leg=false)
#scatter!(t,data)
#plot!(sol)

### Optim Method
using Optim
cost_function = build_optim_objective(prob,tspan,t,data)

result = optimize(cost_function, 0.0, 10.0)
@test 1.5 - result.minimum[1] < 0.01
sol_optimized = solve(prob,tspan)
#plot(sol_optimized,leg=false)
#scatter!(t,data)
#plot!(sol)

result = optimize(cost_function, [1.42], BFGS())
@test 1.5 - result.minimum[1] < 0.01
#sol_optimized2 = solve(prob,tspan)
#plot!(sol_optimized2,leg=false)

#=
### SciPy
model = function (t,p)
  for i in eachindex(f.params)
    setfield!(f,f.params[i],p[i])
  end
  sol = solve(prob,tspan)
  vecout = sol(t)
  y = Matrix{Float64}(length(t),length(prob.u₀))
  for i in 1:length(t)
    y[i,:] = vecout[i]
  end
  vec(y)
end
using PyCall
@pyimport scipy.optimize as opt
opt.curve_fit(model,PyObject(t),PyObject(data))
=#
