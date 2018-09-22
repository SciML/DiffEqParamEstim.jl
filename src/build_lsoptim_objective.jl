export build_lsoptim_objective

function build_lsoptim_objective(prob::DiffEqBase.DEProblem,t,data,alg;
                                 prob_generator = STANDARD_PROB_GENERATOR,
                                 kwargs...)
  vec_data = vec(data)
  data_length = length(vec_data)
  cost_function = function (out,p)
  tmp_prob = prob_generator(prob,p)
  sol = solve(tmp_prob,alg;saveat=t,save_everystep=false,dense=false,kwargs...)
  y = vec(sol)
  y_excess = length(y) - data_length + 1
  y = y[y_excess:length(y)]
  out .= y .- vec_data
  end
end
