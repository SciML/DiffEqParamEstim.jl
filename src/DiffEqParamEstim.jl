module DiffEqParamEstim

using DiffEqBase, LsqFit, OrdinaryDiffEq, LossFunctions, RecursiveArrayTools


  ### LsqFit Method
  function lm_fit(prob::DEProblem,t,data,p0,alg=DefaultODEAlgorithm();kwargs...)
    f = prob.f

    model = function (t,p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,alg)
      y = vecvec_to_mat(sol(t))
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
  end

  function build_optim_objective(prob::DEProblem,t,data,alg=DefaultODEAlgorithm();loss_func = L2DistLoss,kwargs...)
    f = prob.f
    cost_function = function (p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,alg;kwargs...)
      y = vecvec_to_mat(sol(t))
      norm(value(loss_func(),vec(y),vec(data)))
    end
  end

  function build_lsoptim_objective(prob::DEProblem,t,data,alg=DefaultODEAlgorithm();kwargs...)
    cost_function = function (p,out)
      f = (t,u,du) -> prob.f(t,u,du,p)
      uEltype = eltype(p)
      u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
      temp_prob = ODEProblem(f,u0,prob.tspan)
      sol = solve(temp_prob,alg;kwargs...)
      y = vecvec_to_mat(sol(t))
      out[:] = vec(y.-data)
    end
  end

  export lm_fit, build_optim_objective, build_lsoptim_objective

end # module
