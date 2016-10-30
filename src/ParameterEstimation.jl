module ParameterEstimation

using DiffEqBase, LsqFit, OrdinaryDiffEq, LossFunctions


  ### LsqFit Method
  function lm_fit(prob::DEProblem,tspan,t,data,p0;kwargs...)
    f = prob.f

    model = function (t,p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,tspan)
      vecout = sol(t)
      y = Matrix{Float64}(length(t),length(prob.u₀))
      for i in 1:length(t)
        y[i,:] = vecout[i]
      end
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
  end

  function build_optim_objective(prob::DEProblem,tspan,t,data;loss_func = L2DistLoss)
    f = prob.f
    cost_function = function (p)
      for i in eachindex(f.params)
        setfield!(f,f.params[i],p[i])
      end
      sol = solve(prob,tspan)
      vecout = sol(t)
      y = Matrix{Float64}(length(t),length(prob.u₀))
      for i in 1:length(t)
        y[i,:] = vecout[i]
      end
      norm(value(loss_func(),vec(y),vec(data)))
    end
  end
export lm_fit, build_optim_objective

end # module
