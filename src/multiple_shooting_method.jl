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


function multiple_shooting_method(prob::DEProblem,alg,loss,timestamp=nothing;mpg_autodiff = false,
                              verbose_opt = false,verbose_steps = 100,
                              prob_generator = problem_new_parameters,
                              kwargs...)
  if verbose_opt
    count = 0 # keep track of # function evaluations
  end
  multiple_shooting_cost = 0.0
  constraints = []
  # timestamps = sort(sample(2:length(t)-1, 10, replace = false))
  if timestamp == nothing
    length_of_interval = floor(length(t)/10)
    for i in 1:10:
      cost_function = function (p)
        if verbose_opt
          count::Int += 1
          if mod(count,verbose_steps) == 0
            println("f_$count($p)")
          end
        end
        if i==1
          tmp_prob = my_problem_new_parameters(prob,p,prob1.tspan[1],t[length_of_interval],data)
        elseif i==10
          tmp_prob = my_problem_new_parameters(prob,p,t[9*length_of_interval],prob.tspan[2],data)
        else
          tmp_prob = my_problem_new_parameters(prob,p,t[(i-1)*length_of_interval],t[i*length_of_interval],data)
        end
        if typeof(loss) <: Union{CostVData,L2Loss}
          sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
        else
          sol = solve(tmp_prob,alg;kwargs...)
        end
        multiple_shooting_cost +=loss(sol)
        # Contructing the constraints
        if i != 1
          for j in 1:length(sol(t[length_of_interval]))
            push!(constraints, ec[i]=sol(t[(i-1)*length_of_interval])[j])
          end
        end
          ec = sol(t[i*length_of_interval])
      end   #end of cost_function
    end   #end of outer loop
  else
    for i in 1:length(timestamp)
      cost_function = function (p)
        if verbose_opt
          count::Int += 1
          if mod(count,verbose_steps) == 0
            println("f_$count($p)")
          end
        end
        if i==1
          tmp_prob = my_problem_new_parameters(prob,p,prob.tspan[1],timestamp[1],data)
        elseif i==length(timestamp)
          tmp_prob = my_problem_new_parameters(prob,p,timestamp[i],prob.tspan[2],data)
        else
          tmp_prob = my_problem_new_parameters(prob,p,timestamp[i-1],timestamp[i],data)
        end
        if typeof(loss) <: Union{CostVData,L2Loss}
          sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
        else
          sol = solve(tmp_prob,alg;kwargs...)
        end
        multiple_shooting_cost +=loss(sol)
        # Contructing the constraints
        if i != 1
          for j in 1:length(sol(timestamp[1]))
            push!(constraints, ec[i]=sol(timestamp[i-1])[j])
          end
        end
        ec = sol(timestamp[i])
      end
    end
  end   #end of timestamp if condition
  MultipleShootingObjective(multiple_shooting_cost,constraints)
end
