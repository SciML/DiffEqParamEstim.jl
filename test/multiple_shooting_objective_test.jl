using OrdinaryDiffEq, DiffEqParamEstim, BlackBoxOptim, Base.Test, NLopt

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
ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);
                                     discontinuity_weight=0)
function myconstraint(result,x,grad)
  N = length(result)-length(ms_prob.p)
  time_len = Int(floor(length(t)/N))
  time_dur = t[1:time_len]
  sol = []
  for i in 1:length(ms_prob.u0):N
      tmp_prob = remake(ms_prob;u0=x[i:i+length(ms_prob.u0)-1],p=x[N+1:N+length(ms_prob.p)])
      sol=solve(tmp_prob,Tsit5();saveat=time_dur,save_everystep=false,dense=false)
      if (i+1)*time_len < length(t)
        time_dur = t[i*time_len:(i+1)*time_len]
      else
        time_dur = t[i*time_len:Int(length(t))]
      end
      result[i+length(ms_prob.u0)] = sol.u[end][1]
      result[i+(2*length(ms_prob.u0))-1] = sol.u[end][2]
  end
end
# bound = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5),(0.5, 5),(0.5, 10),
                                # (0.5, 5),(0.5, 5),(0.5, 5),(0.5, 5),
                                # (0.5, 5),(0.5, 10),(0.5, 5),(0, 5)]
# bound_lower = AbstractVector[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
# bound_upper = AbstractVector[5,5,5,5,5,5,5,5,5,5,5,5]
# result = bboptimize(ms_obj;SearchRange = bound, MaxSteps = 11e3)
opt = Opt(:LN_COBYLA, 22)
min_objective!(opt, ms_obj.cost_function2)
lower_bounds!(opt,fill(0.0,22))
upper_bounds!(opt,fill(5.0,22))
xtol_rel!(opt,1e-3)
equality_constraint!(opt,myconstraint,fill(1e-3,22))
maxeval!(opt, 10000)
(minf,minx,ret) = NLopt.optimize!(opt,[1.0, 1.0,2.0, 0.20,2.0, 0.20,5.0, 1.0,1.0, 2.0,1.0, 0.0,5.0, 0.0,1.0, 3.0,1.0, 0.0,4.0, 0.0,1.5,1.0])
println(minx)
@test minx[end-1] ≈ 1.5 atol=1e-1
@test minx[end] ≈ 1.0 atol=1e-1
# result = optimize(ms_obj,[0,0,0,0,0,0,0,0,0,0,0,0])
# @test result.archive_output.best_candidate[end-1:end] ≈ [1.5,1.0] atol = 5e-1
# result.archive_output.best_candidate[end-1:end]
# [1.10053, 0.943684, 1.26003, 1.13724, 1.29508, 1.16834, 1.28735, 1.1628, 1.33643, 1.11177, 1.49205, 1.49477]
# [1.12447, 0.935505, 1.25696, 1.16082, 1.23476, 1.27691, 1.24459, 1.13088, 1.44758, 1.07857, 1.18998, 1.06814]
# [1.0, 1.0,2.62167, 0.262287,2.62167, 0.262287,6.87434, 1.72793,0.987225, 2.04868,1.82072, 0.33489,5.96518, 0.583544,1.4558, 3.56928,1.31534, 0.521004,4.30152, 0.309804]