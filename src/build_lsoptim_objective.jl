export build_lsoptim_objective

function build_lsoptim_objective(prob::DEProblem,t,data,alg;kwargs...)
    internal_data = vec(data)
    cost_function = function (p,out)
      f = (t,u,du) -> prob.f(t,u,p,du)
      uEltype = eltype(p)
      u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
      tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
      temp_prob = ODEProblem(f,u0,tspan)
      if alg == nothing
        sol = solve(temp_prob;saveat=t,save_timeseries=false,dense=false,kwargs...)
      else
        sol = solve(temp_prob,alg;saveat=t,save_timeseries=false,dense=false,kwargs...)
      end
      fill_length = length(t)-length(sol)
      for i in 1:fill_length
        push!(sol.u,fill(NaN,size(prob.u0)))
        #push!(sol.u,zeros(prob.u0))
      end
      y = vec(vecvec_to_mat(sol.u))
      out .= y.-internal_data
    end
  end