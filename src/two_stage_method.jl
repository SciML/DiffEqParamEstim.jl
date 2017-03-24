export TwoStageCost, two_stage_method


immutable TwoStageCost{F,D} <: Function
  cost_function::F
  estimated_solution::D 
  estimated_derivative::D
end

(f::TwoStageCost)(p) = f.cost_function(p)


function decide_kernel(kernel)
    if kernel == :Epanechnikov
        return Epanechnikov_kernel
    elseif kernel == :Uniform
        return Uniform_kernel
    else
        return Triangular_kernel
    end
end
function Epanechnikov_kernel(t)
    if abs(t) > 1
        return 0
    else
        return 0.75*(1-t^2)
    end
end

function Uniform_kernel(t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function Triangular_kernel(t)
    if abs(t) > 1
        return 0
    else
        return (1-abs(t))
    end
end


function construct_t1(t,tpoints)
    T1 = []
    for i in 1:length(tpoints)
        push!(T1,[1 tpoints[i]-t])
    end
    foldl(vcat,T1)
    #print(T1)
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


function two_stage_method(prob::DEProblem,tpoints,data,kernel= :Epanechnikov;loss_func = L2DistLoss,kwargs...)
    f = prob.f
    n = length(tpoints)
    h = (n^(-1/5))*(n^(-3/35))*((log(n))^(-1/16))
    estimated_solution = zeros(n,size(data)[2])
    estimated_derivative = zeros(n,size(data)[2])
    kernel_function = decide_kernel(kernel)
    e1 = [1;0]
    e2 = [0;1;0]

    for i in 1:n
        T1 = construct_t1(tpoints[i],tpoints)
        T2 = construct_t2(tpoints[i],tpoints)
        W = construct_w(tpoints[i],tpoints,h,kernel_function)
        estimated_solution[i,:] = e1'*inv(T1'*W*T1)*T1'*W*data
        estimated_derivative[i,:] = e2'*inv(T2'*W*T2)T2'*W*data
    end

    
    # Step - 2
    du = similar(prob.u0)
    cost_function = function (p)
        ff = (t,u,du) -> prob.f(t,u,p,du)
        sol = typeof(prob.u0)[]
        for i in 1:n
          ff(tpoints[i],estimated_solution[i,:],du)
          push!(sol,copy(du))
        end
        out = vecvec_to_mat(sol)
        norm(value(loss_func(),vec(out),vec(estimated_derivative)))
    end
    return TwoStageCost(cost_function, estimated_solution, estimated_derivative)
end
