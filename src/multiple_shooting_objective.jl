export multiple_shooting_objective,merge_solutions

function generate_loss_func(loss,t,i)
  new_loss = nothing
  if typeof(loss)<:L2Loss
    new_loss = L2Loss(t,loss.data[:,i:length(t)])
  end
  new_loss
end

struct Merged_Solution{T1,T2,T3}
  u::T1
  t::T2
  sol::T3
end

function multiple_shooting_objective(prob::DEProblem,alg,loss,regularization=nothing;mpg_autodiff = false,discontinuity_weight=1.0,
                              verbose_opt = false,prob_generator = problem_new_parameters,autodiff_prototype = mpg_autodiff ? zeros(init_N_params) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,
                              kwargs...)
  cost_function = function (p)
    N = length(p)-length(prob.p)
    time_len = Int(floor(length(loss.t)/N))
    time_dur = loss.t[1:time_len]
    sol = []
    loss_val = 0
    for i in 1:length(prob.u0):N
        tmp_prob = remake(prob;u0=p[i:i+length(prob.u0)-1],p=p[N+1:N+length(prob.p)],tspan=(time_dur[1],time_dur[end]))
        if typeof(loss) <: Union{CostVData,L2Loss,LogLikeLoss}
          push!(sol,solve(tmp_prob,alg;saveat=time_dur,save_everystep=false,dense=false,kwargs...))
          if (i+1)*time_len < length(loss.t)
            time_dur = loss.t[i*time_len:(i+1)*time_len]
          else
            time_dur = loss.t[i*time_len:Int(length(loss.t))]
          end
        else
          push!(sol,solve(tmp_prob,alg;kwargs...))
        end
    end
    time_dur = loss.t[1:time_len]
    for i in 2:length(sol)
      loss_val += discontinuity_weight*sum(sol[i][1] - sol[i-1][end])^2
    end
    for i in 1:length(sol)
      new_loss = generate_loss_func(loss,time_dur,1)
      if (i+1)*time_len < length(loss.t)
        time_dur = loss.t[i*time_len:(i+1)*time_len]
      else
        time_dur = loss.t[i*time_len:Int(length(loss.t))]
      end
      if regularization == nothing
        loss_val += new_loss(sol[i])
      else
        loss_val += new_loss(sol[i]) + regularization(p[N:end])
      end
      if verbose_opt
        count::Int += 1
        if mod(count,verbose_steps) == 0
          println("Iteration: $count")
          println("Current Cost: $loss_val")
          println("Parameters: $p")
        end
      end
    end
    loss_val
  end
    if mpg_autodiff
      gcfg = ForwardDiff.GradientConfig(cost_function, autodiff_prototype, autodiff_chunk)
      g! = (x, out) -> ForwardDiff.gradient!(out, cost_function, x, gcfg)
    else
      g! = (x, out) -> Calculus.finite_difference!(cost_function,x,out,:central)
    end
    cost_function2 = function (p,grad)
      if length(grad)>0
          g!(p,grad)
      end
      cost_function(p)
    end
  DiffEqObjective(cost_function,cost_function2)
end

function merge_solutions(prob::DEProblem,alg,t,final_params)
  N = length(final_params)-length(prob.p)
  time_len = Int(floor(length(t)/N))
  time_dur = t[1:time_len]
  sol = []
  for i in 1:length(prob.u0):N
      tmp_prob = remake(prob;u0=final_params[i:i+length(prob.u0)-1],p=final_params[N+1:N+length(prob.p)],tspan=(time_dur[1],time_dur[end]))
      push!(sol,solve(tmp_prob,alg;saveat=time_dur,save_everystep=false,dense=false))
      if (i+1)*time_len < length(t)
        time_dur = t[i*time_len:(i+1)*time_len]
      else
        time_dur = t[i*time_len:Int(length(t))]
      end
  end
  u = [i for j in 1:length(sol) for i in sol[j].u]
  t = [i for j in 1:length(sol) for i in sol[j].t]
  Merged_Solution(u,t,sol)
end