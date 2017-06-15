export DiffEqMaximumlikelihood, build_maximum_likelihood_objective

immutable DiffEqMaximumlikelihood{F,F2} <: Function
  maximum_likelihood_function::F
  maximum_likelihood_function2::F2
end

(f::DiffEqMaximumlikelihood)(x) = f.maximum_likelihood_function(x)
(f::DiffEqMaximumlikelihood)(x,y) = f.maximum_likelihood_function2(x,y)

function build_maximum_likelihood_objective(prob::DEProblem,alg,loss;mpg_autodiff = false, verbose_opt = false,verbose_steps = 100, prob_generator = problem_new_parameters, kwargs...)
  cost_function = build_loss_objective(prob::DEProblem,alg,loss;mpg_autodiff = false, verbose_opt = false,verbose_steps = 100, prob_generator = problem_new_parameters, kwargs...)
  DiffEqMaximumlikelihood(cost_function.cost_function,cost_function.cost_function2)
end
