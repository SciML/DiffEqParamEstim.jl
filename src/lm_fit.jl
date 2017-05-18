### LsqFit Method
export lm_fit

function lm_fit(prob::DEProblem,t,data,p0,alg;
                prob_generator = problem_new_parameters,
                kwargs...)
    f = prob.f
    model = function (t,p)
      tmp_prob = prob_generator(prob,p)
      sol = solve(tmp_prob,alg;saveat=t,save_everystep=false,dense=false,kwargs...)
      y = vecvec_to_mat(sol.u)
      vec(y)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
end
