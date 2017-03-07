### LsqFit Method
export lm_fit
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