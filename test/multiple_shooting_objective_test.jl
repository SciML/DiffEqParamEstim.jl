using OrdinaryDiffEq, DiffEqParamEstim, BlackBoxOptim, Base.Test

pf_func = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3.0 * u[2] + u[1]*u[2]
end
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0]
prob1 = ODEProblem(pf_func,u0,tspan,p)
t = collect(linspace(0,10,200))
sol = solve(prob1,Tsit5())
obj = multiple_shooting_objective(prob1,Tsit5(),L2Loss(t,sol),[0,0,1,1,2,2,3,3,4,4,5,5,1.2,1.2];discontinuity_weight=2.8)
bound = Tuple{Float64, Float64}[(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10)]
result = bboptimize(obj;SearchRange = bound, MaxSteps = 11e3)

@test result.archive_output.best_candidate[end-1:end] ≈ [1.5,1.0] atol = 1e-1