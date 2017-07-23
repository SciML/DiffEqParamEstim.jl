export TwoStageCost, two_stage_method


immutable TwoStageCost{F,F2,D} <: Function
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
    elseif kernel == :TriTriangular
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
    T1 = []
    for i in 1:length(tpoints)
        push!(T1,[1 tpoints[i]-t])
    end
    foldl(vcat,T1)
end
function construct_t2(t,tpoints)
    T2 = []
    for i in 1:length(tpoints)
        push!(T2,[1 tpoints[i]-t (tpoints[i]-t)^2])
    end
    foldl(vcat,T2)
end
function construct_w(t,tpoints,h,kernel_function)
    n = length(tpoints)
    W = zeros(n)
    for i in 1:n
        W[i] = kernel_function((tpoints[i]-t)/h)/h
    end
    diagm(W)
end
function construct_estimated_solution_and_derivative!(estimated_solution,estimated_derivative,e1,e2,data,kernel_function,tpoints,h,n)
  for i in 1:n
      T1 = construct_t1(tpoints[i],tpoints)
      T2 = construct_t2(tpoints[i],tpoints)
      W = construct_w(tpoints[i],tpoints,h,kernel_function)
      estimated_solution[i,:] = e1'*inv(T1'*W*T1)*T1'*W*data'
      estimated_derivative[i,:] = e2'*inv(T2'*W*T2)T2'*W*data'
  end
end

function two_stage_method(prob::DEProblem,tpoints,data;kernel= :Epanechnikov,
                          loss_func = L2DistLoss,mpg_autodiff = false,
                          verbose = false,verbose_steps = 100)
    f = prob.f
    n = length(tpoints)
    h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))
    estimated_solution = zeros(n,size(data)[1])
    estimated_derivative = zeros(n,size(data)[1])
    kernel_function = decide_kernel(kernel)
    e1 = [1;0]
    e2 = [0;1;0]
    construct_estimated_solution_and_derivative!(estimated_solution,estimated_derivative,e1,e2,data,kernel_function,tpoints,h,n)


    # Step - 2
    du = similar(prob.u0)
    cost_function = function (p)
        ff = (t,u,du) -> prob.f(t,u,p,du)
        sol = Vector{typeof(prob.u0)}(n)
        for i in 1:n
          ff(tpoints[i],estimated_solution[i,:],du)
          sol[i] = copy(du)
        end
        norm(value(loss_func(),vec(estimated_derivative'),vec(VectorOfArray(sol))))
    end

    if mpg_autodiff
      gcfg = ForwardDiff.GradientConfig(zeros(length(f.syms)))
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
      if verbose
        count::Int += 1
        if mod(count,verbose_steps) == 0
          println("f_$count($p)")
        end
      end
      cost_function(p)
    end

    return TwoStageCost(cost_function, cost_function2, estimated_solution, estimated_derivative)
end
