module DiffEqParamEstim

using DiffEqBase, LsqFit, LossFunctions, RecursiveArrayTools


  ### LsqFit Method
  function lm_fit(prob::DEProblem,t,data,p0,alg;kwargs...)
    f = prob.f

    model = function (t,p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,alg;saveat=t,save_timeseries=false,dense=false,kwargs...)
      y = vecvec_to_mat(sol.u)
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
  end

  function build_optim_objective(prob::DEProblem,t,data,alg;loss_func = L2DistLoss(),kwargs...)
    f = prob.f
    cost_function = function (p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end

      sol = solve(prob,alg;saveat=t,save_timeseries=false,dense=false,kwargs...)
      fill_length = length(t)-length(sol)
      for i in 1:fill_length
        push!(sol.u,fill(NaN,size(prob.u0)))
        #push!(sol.u,zeros(prob.u0))
      end
      y = vec(vecvec_to_mat(sol.u))
      norm(value(loss_func,vec(y),vec(data)))
    end
  end

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

  export lm_fit, build_optim_objective, build_lsoptim_objective

end # module
