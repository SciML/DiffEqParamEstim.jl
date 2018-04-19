using OrdinaryDiffEq, DiffEqParamEstim, BlackBoxOptim, Base.Test

ms_f = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3.0 * u[2] + u[1]*u[2]
end
ms_u0 = [1.0;1.0]
tspan = (0.0,10.0)
ms_p = [1.5,1.0]
ms_prob = ODEProblem(ms_f,ms_u0,tspan,ms_p)
t = collect(linspace(0,10,200))
data = Array(solve(ms_prob,Tsit5(),saveat=t))
ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data),
                                     [0,0,1,1,2,2,3,3,4,4,5,5,1.2,1.2];
                                     discontinuity_weight=2.8)
bound = Tuple{Float64, Float64}[(0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),
                                (0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),
                                (0.5, 10),(0.5, 10),(0.5, 10),(0.5, 10),
                                (0.5, 10),(0.5, 10)]
result = bboptimize(ms_obj;SearchRange = bound, MaxSteps = 11e3)

@test result.archive_output.best_candidate[end-1:end] ≈ [1.5,1.0] atol = 1e-1
result.archive_output.best_candidate[end-1:end]
