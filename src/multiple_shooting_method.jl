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
    if verbose_opt
      count::Int += 1
      if mod(count,verbose_steps) == 0
        println("f_$count($p)")
      end
    end
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
  boundary_condition = nothing
  if timestamp == nothing
    length_of_interval = floor(length(t)/10)
    for i in 1:10:
      if i==1
        construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,prob.tspan[1],t[length_of_interval],data)
      elseif i==10
        construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,t[9*length_of_interval],prob.tspan[2],data)
      else
        construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,t[(i-1)*length_of_interval],t[i*length_of_interval],data)
      end
    end   #end of for loop
  else
    for i in 1:length(timestamp)
        if i==1
          construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,prob.tspan[1],timestamp[1],data)
        elseif i==length(timestamp)
          construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,timestamp[i],prob.tspan[2],data)
        else
          construct_objective_constraints!(multiple_shooting_cost,constraints,boundary_condition,prob,timestamp[i-1],timestamp[i],data)
        end
      end  #end of for loop
  end   #end of timestamp if condition
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
    if timestamp
      for i in 1:(timestamp-1)*length(prob.u0)
        f! = (x, out) -> ForwardDiff.gradient!(out, constraints[i], x, gcfg)
      end
    else
      for i in 1:9*length(prob.u0)
        f! = (x, out) -> ForwardDiff.gradient!(out, constraints[i], x, gcfg)
      end
    end
  else
    g! = (x, out) -> Calculus.finite_difference!(multiple_shooting_cost,x,out,:central)
    if timestamp
      for i in 1:(timestamp-1)*length(prob.u0)
        f! = (x, out) -> Calculus.finite_difference!(constraints[i],x,out,:central)
      end
    else
      for i in 1:9*length(prob.u0)
        f! = (x, out) -> Calculus.finite_difference!(constraints[i],x,out,:central)
      end
    end
  end
  MultipleShootingObjective(multiple_shooting_cost,constraints)
end
