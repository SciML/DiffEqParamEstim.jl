module DiffEqParamEstim

using DiffEqBase, LsqFit, OrdinaryDiffEq, LossFunctions, RecursiveArrayTools


  ### LsqFit Method
  function lm_fit(prob::DEProblem,tspan,t,data,p0;kwargs...)
    f = prob.f

    model = function (t,p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,tspan)
      y = vecvec_to_mat(sol(t))
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
  end

  function build_optim_objective(prob::DEProblem,tspan,t,data;loss_func = L2DistLoss,kwargs...)
    f = prob.f
    cost_function = function (p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,tspan;kwargs...)
      y = vecvec_to_mat(sol(t))
      norm(value(loss_func(),vec(y),vec(data)))
    end
  end

  function build_lsoptim_objective(prob::DEProblem,tspan,t,data;kwargs...)
    cost_function = function (p,out)
      f = (t,u,du) -> prob.f(t,u,du,p)
      uEltype = eltype(p)
      u₀ = [uEltype(prob.u₀[i]) for i in 1:length(prob.u₀)]
      temp_prob = ODEProblem(f,u₀)
      sol = solve(temp_prob,tspan;kwargs...)
      y = vecvec_to_mat(sol(t))
      out[:] = vec(y.-data)
    end
  end

  export lm_fit, build_optim_objective, build_lsoptim_objective

end # module
