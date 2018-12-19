export TwoStageCost, two_stage_method


struct TwoStageCost{F,F2,D} <: Function
  cost_function::F
  cost_function2::F2
  estimated_solution::D
  estimated_derivative::D
end

struct MyTag end
struct DiffCache{T<:AbstractArray, S<:AbstractArray}
    du::T
    dual_du::S
end

function DiffCache(T, size, ::Type{Val{chunk_size}}) where chunk_size
    DiffCache(zeros(eltype(T), size...), zeros(eltype(ForwardDiff.Dual{nothing,T,chunk_size}), size...))
end

DiffCache(u::AbstractArray) = DiffCache(eltype(u),size(u),Val{ForwardDiff.pickchunksize(length(u))})

get_tmp(dc::DiffCache, ::Type{T}) where T<:ForwardDiff.Dual = dc.dual_du
get_tmp(dc::DiffCache, T) = dc.du

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
    Matrix(Diagonal(W))
end
function construct_estimated_solution_and_derivative!(estimated_solution,estimated_derivative,e1,e2,data,kernel_function,tpoints,h,n)
  for i in 1:n
      T1 = construct_t1(tpoints[i],tpoints)
      T2 = construct_t2(tpoints[i],tpoints)
      W = construct_w(tpoints[i],tpoints,h,kernel_function)
      estimated_solution[:,i] = e1'*inv(T1'*W*T1)*T1'*W*data'
      estimated_derivative[:,i] = e2'*inv(T2'*W*T2)T2'*W*data'
  end
end

function two_stage_method(prob::DiffEqBase.DEProblem,tpoints,data;kernel= :Epanechnikov,
                          loss_func = L2Loss,mpg_autodiff = false,
                          verbose = false,verbose_steps = 100,
                          autodiff_prototype = mpg_autodiff ? zeros(length(prob.p)) : nothing,
                          autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing)
    f = prob.f
    n = length(tpoints)
    h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))
    estimated_solution = zeros(size(data)[1],n)
    estimated_derivative = zeros(size(data)[1],n)
    kernel_function = decide_kernel(kernel)
    e1 = [1;0]
    e2 = [0;1;0]
    du = similar(prob.u0) # have to adjust type for autodifferentiation
    sol = Vector{typeof(du)}(undef,n)
    if mpg_autodiff
      du_cache = DiffCache(du)
      sol_cache = Vector{typeof(du_cache)}(undef,n)
    end
    construct_estimated_solution_and_derivative!(estimated_solution,estimated_derivative,e1,e2,data,kernel_function,tpoints,h,n)
    # Step - 2
    cost_function = function (p)
        if mpg_autodiff
          du_ = get_tmp(du_cache,eltype(p))
          du = reinterpret(eltype(p),du_)
          sol_ = map(i -> get_tmp(i,eltype(p)),sol_cache)
          sol = map(i -> reinterpret(eltype(p),i),sol_)
        end 
        f = prob.f
        for i in 1:n
          est_sol = @view estimated_solution[:,i]
          f(du,est_sol,p,tpoints[i])
          sol[i] = copy(du)
        end
        sum(abs2(vec(estimated_derivative) .- vec(VectorOfArray(sol))))
    end

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
