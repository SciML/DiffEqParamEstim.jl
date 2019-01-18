export multiple_shooting_objective

function generate_loss_func(loss,t,i)
  new_loss = nothing
  if typeof(loss)<:L2Loss && i==1
    new_loss = L2Loss(t,loss.data[:,i:i+length(t)])
  elseif typeof(loss)<:L2Loss
    i = (i-1)*length(t)
    if i+length(t) < size(loss.data)[2]
      new_loss = L2Loss(t,loss.data[:,i:i+length(t)])
    else
      new_loss = L2Loss(t,loss.data[:,i:size(loss.data)[2]])
    end
  end
  new_loss
end

struct Merged_Solution{T1,T2,T3}
  u::T1
  t::T2
  sol::T3
end

function multiple_shooting_objective(prob::DiffEqBase.DEProblem,alg,loss,
                              regularization=nothing;prior=nothing,
                              mpg_autodiff = false,discontinuity_weight=1.0,
                              verbose_opt = false,
                              prob_generator = STANDARD_PROB_GENERATOR,
                              autodiff_prototype = mpg_autodiff ? zeros(init_N_params) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,
                              kwargs...)
  cost_function = function (p)
    N = length(p)-length(prob.p)
    time_len = Int(floor((length(loss.t)*length(prob.u0))/N))
    time_dur = loss.t[1:time_len+1]
    sol = []
    loss_val = 0
    j = 1
    for i in 1:length(prob.u0):N
        tmp_prob = remake(prob;u0=p[i:i+length(prob.u0)-1],p=p[N+1:N+length(prob.p)],tspan=(time_dur[1],time_dur[end]))
        if typeof(loss) <: Union{L2Loss,LogLikeLoss}
          push!(sol,solve(tmp_prob,alg;saveat=time_dur,save_everystep=false,dense=false,kwargs...))
          if (j+1)*time_len < length(loss.t)
            time_dur = loss.t[j*time_len+1:(j+1)*time_len+1]
          else
            time_dur = loss.t[j*time_len+1:Int(length(loss.t))]
          end
        else
          push!(sol,solve(tmp_prob,alg;kwargs...))
        end
        j = j+1
    end
    if any((s.retcode != :Success for s in sol))
        return Inf
    end
    u = [i for j in 1:length(sol) for i in sol[j].u[1:end-1]]
    t = [i for j in 1:length(sol) for i in sol[j].t[1:end-1]]
    push!(u,sol[end].u[end])
    push!(t,sol[end].t[end])
    sol_loss = Merged_Solution(u,t,sol)
    sol_new = DiffEqBase.build_solution(prob,alg,loss.t,sol_loss.u,
                                        retcode = :Success)
    loss_val = loss(sol_new)
    if prior != nothing
      loss_val += prior_loss(prior,p[end-length(prior):end])
    end
    if regularization != nothing
      loss_val += regularization(p)
    end
    for i in 2:length(sol)
      loss_val += discontinuity_weight*sum(sol[i][1] - sol[i-1][end])^2
    end
    if verbose_opt
      count::Int += 1
      if mod(count,verbose_steps) == 0
        println("Iteration: $count")
        println("Current Cost: $loss_val")
        println("Parameters: $p")
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
