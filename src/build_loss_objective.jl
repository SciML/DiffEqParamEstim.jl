export DiffEqObjective, build_loss_objective

struct DiffEqObjective{F,F2} <: Function
  cost_function::F
  cost_function2::F2
end

function diffeq_sen_full(f, u0, tspan, p, t, alg; kwargs...)
  prob = ODELocalSensitivityProblem(f,u0,tspan,p)
  sol = solve(prob,alg;kwargs...)(t)
  nvar = length(u0)
  sol[1:nvar,:], [sol[i*nvar+1:i*nvar+nvar,:] for i in 1:length(p)]
end

function diffeq_sen_l2!(res, df, u0, tspan, p, t, data, alg;kwargs...)
  prob = ODEProblem(df,u0,tspan,p)
  sol = solve(prob, alg, saveat=t; kwargs...)
  function dg(out,u,p,t,i)
    @. out = 2 * (data[:,i] - u)
  end
  res .= adjoint_sensitivities(sol,alg,dg,t,kwargs...)[1,:]
end

(f::DiffEqObjective)(x) = f.cost_function(x)
(f::DiffEqObjective)(x,y) = f.cost_function2(x,y)

function build_loss_objective(prob::DiffEqBase.DEProblem,alg,loss,regularization=nothing;
                              prior=nothing,mpg_autodiff = false,
                              verbose_opt = false,verbose_steps = 100,
                              prob_generator = STANDARD_PROB_GENERATOR,
                              autodiff_prototype = mpg_autodiff ? zero(prob.p) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,flsa_gradient = false,adjsa_gradient= false,
                              kwargs...)
  if verbose_opt
    count = 0 # keep track of # function evaluations
  end
  cost_function = function (p)
    tmp_prob = prob_generator(prob,p)
    if typeof(loss) <: Union{L2Loss,LogLikeLoss}
      sol = solve(tmp_prob,alg;saveat=loss.t,save_everystep=false,dense=false,kwargs...)
    else
      sol = solve(tmp_prob,alg;kwargs...)
    end

    loss_val = loss(sol)

    if prior != nothing
      loss_val += prior_loss(prior,p)
    end
    if regularization != nothing
      loss_val += regularization(p)
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
  elseif flsa_gradient
    if typeof(loss) <: L2Loss
      function g!(x,out)
        sol_,sens = diffeq_sen_full(prob.f,prob.u0,prob.tspan,x,loss.t,alg)
        l2lossgradient!(out,sol_,loss.data,sens,length(prob.p))
      end
    else
      throw("LSA gradient only for L2Loss")
    end
  elseif adjsa_gradient
    g! = (x,out) -> diffeq_sen_l2!(out,prob.f,prob.u0,prob.tspan,x,loss.t,loss.data,alg)
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
