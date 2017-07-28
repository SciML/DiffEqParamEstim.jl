export MultipleShootingObjective, multiple_shooting_objective

immutable MultipleShootingObjective{F,C} <: Function
  multiple_shooting_cost_function::F
  constraints::C
end

my_problem_new_parameters = function (prob::ODEProblem,p,t1,t2,data)
  f = (t,u,du) -> prob.f(t,u,p,du)
  uEltype = eltype(p)
  tspan = (uEltype(t1),uEltype(t2))
  u0 = [uEltype(data[:,1][i]) for i in 1:length(prob.u0)]
  ODEProblem(prob.f,u0,(t1,t2))
end


function multiple_shooting_objective(prob::DEProblem,alg,loss,timestamp=nothing;mpg_autodiff = false,
                              verbose_opt = false,verbose_steps = 100,
                              prob_generator = problem_new_parameters,
                              kwargs...)
  if verbose_opt
    count = 0 # keep track of # function evaluations
  end
  timestamps = sort(sample(2:length(t)-1, 10, replace = false))
  multiple_shooting_cost_function = 0.0
  constraints = []
  for i in 1:length(timestamps):
    cost_function = function (p)
      if verbose_opt
        count::Int += 1
        if mod(count,verbose_steps) == 0
          println("f_$count($p)")
        end
      end
      tmp_prob = my_problem_new_parameters(prob,p)
      if typeof(loss) <: Union{CostVData,L2Loss}
        sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
      else
        sol = solve(tmp_prob,alg;kwargs...)
      end
      loss(sol)
      multiple_shooting_cost_function +=loss(sol)
      for j in 1:length(sol(timestamps[i]))
        push!(constraints, ec[i]=sol(timestamps[i])[j])
      end
        ec = sol(timestamps[i])
    end
  end
  MultipleShootingObjective(multiple_shooting_cost_function,cost_function2)
end
