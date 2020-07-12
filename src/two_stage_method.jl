export TwoStageCost, two_stage_method


struct TwoStageCost{F,F2,D} <: Function
  cost_function::F
  cost_function2::F2
  estimated_solution::D
  estimated_derivative::D
end

(f::TwoStageCost)(p) = f.cost_function(p)
(f::TwoStageCost)(p,g) = f.cost_function2(p,g)

function decide_kernel(kernel)
    if kernel == :Epanechnikov
        return Epanechnikov_kernel
    elseif kernel == :Uniform
        return Uniform_kernel
    elseif kernel == :Triangular
        return Triangular_kernel
    elseif kernel == :Quartic
      return Quartic_Kernel
    elseif kernel == :Triweight
      return Triweight_Kernel
    elseif kernel == :Tricube
      return Tricube_Kernel
    elseif kernel == :Gaussian
      return Gaussian_Kernel
    elseif kernel == :Cosine
      return Cosine_Kernel
    elseif kernel == :Logistic
      return Logistic_Kernel
    elseif kernel == :Sigmoid
      return Sigmoid_Kernel
    else
      return Silverman_Kernel
    end
end


function construct_t1(t,tpoints)
    mapreduce(vcat,1:length(tpoints)) do i
        [1 tpoints[i]-t]
    end
end
function construct_t2(t,tpoints)
    mapreduce(vcat,1:length(tpoints)) do i
        [1 tpoints[i]-t (tpoints[i]-t)^2]
    end
end
function construct_w(t,tpoints,h,kernel_function)
    W = @. kernel_function((tpoints-t)/h)/h
    Matrix(Diagonal(W))
end
function construct_estimated_solution_and_derivative!(data,kernel_function,tpoints)
  _one = oneunit(first(data))
  _zero = zero(first(data))
  e1 = [_one;_zero]
  e2 = [_zero;_one;_zero]
  n = length(tpoints)
  h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))

  x = map(tpoints) do _t
      T1 = construct_t1(_t,tpoints)
      T2 = construct_t2(_t,tpoints)
      W = construct_w(_t,tpoints,h,kernel_function)
      e2'*inv(T2'*W*T2)T2'*W*data',e1'*inv(T1'*W*T1)*T1'*W*data'
  end
  estimated_derivative = reduce(hcat,transpose.(first.(x)))
  estimated_solution = reduce(hcat,transpose.(last.(x)))
  estimated_derivative,estimated_solution
end

function construct_cost_function(f,du,preview_est_sol,preview_est_deriv,tpoints)
  function (p)
      _du = DiffEqBase.get_tmp(du,p)
      vecdu = vec(_du)
      cost = zero(first(p))
      for i in 1:length(preview_est_sol)
        est_sol = preview_est_sol[i]
        f(_du,est_sol,p,tpoints[i])
        vecdu .= vec(preview_est_deriv[i]) .- vec(_du)
        cost += sum(abs2,vecdu)
      end
      sqrt(cost)
  end
end

function two_stage_method(prob::DiffEqBase.DEProblem,tpoints,data;kernel= :Epanechnikov,
                          loss_func = L2Loss,mpg_autodiff = false,
                          verbose = false,verbose_steps = 100,
                          autodiff_chunk = Val{ForwardDiff.pickchunksize(length(prob.p))})
    f = prob.f
    kernel_function = decide_kernel(kernel)
    estimated_derivative,estimated_solution = construct_estimated_solution_and_derivative!(data,kernel_function,tpoints)

    # Step - 2

    du = DiffEqBase.dualcache(similar(prob.u0), autodiff_chunk)
    preview_est_sol = [@view estimated_solution[:,i] for i in 1:size(estimated_solution,2)]
    preview_est_deriv = [@view estimated_derivative[:,i] for i in 1:size(estimated_solution,2)]
    cost_function = construct_cost_function(f,du,preview_est_sol,preview_est_deriv,tpoints)

    if mpg_autodiff
      gcfg = ForwardDiff.GradientConfig(cost_function, autodiff_prototype, autodiff_chunk)
      g! = (x, out) -> ForwardDiff.gradient!(out, cost_function, x, gcfg)
    else
      g! = (x, out) -> Calculus.finite_difference!(cost_function,x,out,:central)
    end
    if verbose
      count = 0 # keep track of # function evaluations
    end
    cost_function2 = function (p,grad)
      if length(grad)>0
        g!(p,grad)
      end
      loss_val = cost_function(p)
      if verbose
        count::Int += 1
        if mod(count,verbose_steps) == 0
          println("Iteration: $count")
          println("Current Cost: $loss_val")
          println("Parameters: $p")
        end
      end
      loss_val
    end

    return TwoStageCost(cost_function, cost_function2, estimated_solution, estimated_derivative)
end
