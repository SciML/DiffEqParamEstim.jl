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
export lm_fit, build_optim_objective

end # module
