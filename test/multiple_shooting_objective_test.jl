using OrdinaryDiffEq, DiffEqParamEstim, Test, NLopt, BlackBoxOptim
ms_f = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3.0 * u[2] + u[1]*u[2]
end
ms_u0 = [1.0;1.0]
tspan = (0.0,10.0)
ms_p = [1.5,1.0]
ms_prob = ODEProblem(ms_f,ms_u0,tspan,ms_p)
t = collect(range(0, stop=10, length=200))
data = Array(solve(ms_prob,Tsit5(),saveat=t,abstol=1e-12,reltol=1e-12))
bound = Tuple{Float64, Float64}[(0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),(0, 10),(0, 10)]


ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);discontinuity_weight=1.0,abstol=1e-12,reltol=1e-12)
result = bboptimize(ms_obj;SearchRange = bound, MaxSteps = 21e3)
@test result.archive_output.best_candidate[end-1:end] ≈ [1.5,1.0] atol = 2e-1

priors = [Truncated(Normal(1.5,0.1),0,2),Truncated(Normal(1.0,0.1),0,1.5)]
ms_obj1 = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);priors=priors,discontinuity_weight=1.0,abstol=1e-12,reltol=1e-12)
result = bboptimize(ms_obj1;SearchRange = bound, MaxSteps = 21e3)
@test result.archive_output.best_candidate[end-1:end] ≈ [1.5,1.0] atol = 2e-1