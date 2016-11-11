module DiffEqParamEstim

using DiffEqBase, LsqFit, OrdinaryDiffEq, LossFunctions, RecursiveArrayTools


  ### LsqFit Method
  function lm_fit(prob::DEProblem,t,data,p0,alg=nothing;kwargs...)
    f = prob.f

    model = function (t,p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      if alg == nothing
        sol = solve(prob)
      else
        sol = solve(prob,alg)
      end
      y = vecvec_to_mat(sol(t))
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
  end

  function build_optim_objective(prob::DEProblem,t,data,alg=nothing;loss_func = L2DistLoss,kwargs...)
    f = prob.f
    cost_function = function (p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end

      if alg == nothing
        sol = solve(prob;kwargs...)
      else
        sol = solve(prob,alg;kwargs...)
      end

      y = vecvec_to_mat(sol(t))
      norm(value(loss_func(),vec(y),vec(data)))
    end
  end

  function build_lsoptim_objective(prob::DEProblem,t,data,alg=nothing;kwargs...)
    cost_function = function (p,out)
      f = (t,u,du) -> prob.f(t,u,du,p)
      uEltype = eltype(p)
      u0 = [uEltype(prob.u0[i]) for i in 1:length(prob.u0)]
      tspan = (uEltype(prob.tspan[1]),uEltype(prob.tspan[2]))
      temp_prob = ODEProblem(f,u0,tspan)
      if alg == nothing
        sol = solve(temp_prob;kwargs...)
      else
        sol = solve(temp_prob,alg;kwargs...)
      end
      y = vecvec_to_mat(sol(t))
      out[:] = vec(y.-data)
    end
  end

  export lm_fit, build_optim_objective, build_lsoptim_objective

end # module
