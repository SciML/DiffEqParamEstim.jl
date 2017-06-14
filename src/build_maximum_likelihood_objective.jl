export DiffEqMaximumLikelihood, build_maximum_likelihood_objective

immutable DiffEqMaximumLikelihood{F,F2} <: Function
  maximum_likelihood_function::F
  maximum_likelihood_function2::F2
end

(f::DiffEqMaximumLikelihood)(x) = f.maximum_likelihood_function(x)
(f::DiffEqMaximumLikelihood)(x,y) = f.maximum_likelihood_function2(x,y)

function build_maximum_likelihood_objective(prob::DEProblem,alg,likelihood;mpg_autodiff = false, verbose_opt = false,verbose_steps = 100, prob_generator = problem_new_parameters, kwargs...)
  if verbose_opt
    count = 0 # keep track of # function evaluations
  end
  maximum_likelihood_function = function (p)
    if verbose_opt
      count::Int += 1
      if mod(count,verbose_steps) == 0
        println("f_$count($p)")
      end
    end
    tmp_prob = prob_generator(prob,p)
    sol = solve(tmp_prob,alg;saveat=likelihood.t,save_everystep=false,dense=false,kwargs...)
    likelihood(sol)
  end

  if mpg_autodiff
    gcfg = ForwardDiff.GradientConfig(zeros(num_params(prob)))
    g! = (x, out) -> ForwardDiff.gradient!(out, cost_function, x, gcfg)
  else
    g! = (x, out) -> Calculus.finite_difference!(cost_function,x,out,:central)
  end
  maximum_likelihood_function2 = function (p,grad)
    if length(grad)>0
      g!(p,grad)
    end
    maximum_likelihood_function(p)
  end
  DiffEqMaximumLikelihood(maximum_likelihood_function,maximum_likelihood_function2)



end
