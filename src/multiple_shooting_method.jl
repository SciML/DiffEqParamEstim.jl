export MultipleShootingObjective, multiple_shooting_method

immutable MultipleShootingObjective{F,C} <: Function
  multiple_shooting_cost::F
  constraints::C
end

my_problem_new_parameters = function (prob::ODEProblem,p,t1,t2,data)
  f = (t,u,du) -> prob.f(t,u,p,du)
  uEltype = eltype(p)
  tspan = (uEltype(t1),uEltype(t2))
  u0 = [uEltype(data[:,t1][i]) for i in 1:length(prob.u0)]
  ODEProblem(prob.f,u0,(t1,t2))
end

function construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob::ODEProblem,t1,t2,data)
  cost_function = function (p)
    tmp_prob = my_problem_new_parameters(prob,p,t1,t2,data)
    if typeof(loss) <: Union{CostVData,L2Loss}
      sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
    else
      sol = solve(tmp_prob,alg;kwargs...)
    end
    multiple_shooting_cost +=loss(sol)
    # Construct the constraints
    if boundary_condition != nothing
      for j in 1:length(sol(t1))
        push!(constraints, boundary_condition[j]-sol(t1)[j])
      end
    end
    boundary_condition = sol(t2)
  end  #end of cost_function function
end #end of construct_objective_constraints

function multiple_shooting_method(prob::DEProblem,alg,loss,timestamp=nothing;mpg_autodiff = false,
                              verbose_opt = false,verbose_steps = 100,
                              prob_generator = problem_new_parameters,
                              kwargs...)
  if verbose_opt
    count = 0 # keep track of # function evaluations
  end
  multiple_shooting_cost = 0.0
  constraints = []
  data = loss.data
  boundary_condition = nothing
  if timestamp == nothing
    timestamp = collect(linspace(prob.tspan[1],prob.tspan[2],10))
  end
  for i in 1:length(timestamp)-1
    construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,timestamp[i],timestamp[i+1],data)
  end  #end of for loop
  if verbose_opt
    count::Int += 1
    if mod(count,verbose_steps) == 0
      println("Iteration: $count")
      println("Current Cost: $multiple_shooting_cost")
      println("Parameters: $p")
    end
  end
  if mpg_autodiff
    gcfg = ForwardDiff.GradientConfig(zeros(num_params(prob)))
    g! = (x, out) -> ForwardDiff.gradient!(out, multiple_shooting_cost, x, gcfg)
    for i in 1:(length(timestamp)-1)*length(prob.u0)
      f! = (x, out) -> ForwardDiff.gradient!(out, constraints[i], x, gcfg)
    end
  else
    g! = (x, out) -> Calculus.finite_difference!(multiple_shooting_cost,x,out,:central)
    for i in 1:(length(timestamp)-1)*length(prob.u0)
      f! = (x, out) -> Calculus.finite_difference!(constraints[i],x,out,:central)
    end
  end #end of if condition
  MultipleShootingObjective(multiple_shooting_cost,constraints)
end
