### LsqFit Method
export lm_fit

function lm_fit(prob::DEProblem,t,data,p0,alg;kwargs...)
    f = prob.f
    model = function (t,p)
      tmp_prob = problem_new_parameters(prob,p)
      if alg == nothing
        sol = solve(tmp_prob;saveat=t,save_everystep=false,dense=false,kwargs...)
      else
        sol = solve(tmp_prob,alg;saveat=t,save_everystep=false,dense=false,kwargs...)
      end
      y = vecvec_to_mat(sol.u)
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
end
