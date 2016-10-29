using ParameterEstimation, OrdinaryDiffEq, ParameterizedFunctions, LsqFit
using Base.Test

f = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0

u0 = [1.0;1.0]
tspan = [0;10.0]

prob = ODEProblem(f,u0)
sol = solve(prob,tspan)

using Plots
gr()
plot(sol)

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]

plotseries = Matrix{Float64}(length(randomized),length(u0))
for i in 1:length(randomized)
  plotseries[i,:] = randomized[i]
end
scatter!(t,plotseries)

scatter(t,plotseries)

f = LotkaVolterraTest(a=1.42)
prob = ODEProblem(f,u0)
sol = solve(prob,tspan)
plot!(sol)

function model(t,p)
  for i in eachindex(f.params)
    setfield!(f,f.params[i],p[i])
  end
  sol = solve(prob,tspan)
  vecout = sol(t)
  y = Matrix{Float64}(length(t),length(u0))
  for i in 1:length(t)
    y[i,:] = vecout[i]
  end
  vec(y)
end

fit = curve_fit(model,t,vec(plotseries),[1.42],show_trace=true,lambda=10000.0)

#param = [3.0;5.0;5.0]
param = fit.param
for i in eachindex(f.params)
  setfield!(f,f.params[i],param[i])
end
#prob = ODEProblem(f,u0)
sol_optimized = solve(prob,tspan)
plot(sol_optimized,leg=false)
scatter!(t,plotseries)
plot!(sol)
