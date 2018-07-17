### LsqFit Method
export lm_fit

function lm_fit(prob::DiffEqBase.DEProblem,t,data,p0,alg;
                prob_generator = remake,
                kwargs...)
    f = prob.f
    model = function (t,p)
      tmp_prob = prob_generator(prob;p=p)
      sol = solve(tmp_prob,alg;saveat=t,save_everystep=false,dense=false,kwargs...)
      vec(sol)
    end
    curve_fit(model,t,vec(data),p0;kwargs...)
end
