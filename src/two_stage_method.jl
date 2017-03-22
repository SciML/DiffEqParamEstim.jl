export two_stage_method

# Step - 1
function decide_kernel(kernel)
    if kernel == "Epanechnikov"
        return Epanechnikov_kernel
    elseif kernel == "Uniform"
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


function two_stage_method(prob::DEProblem,tpoints,data,kernel="Epanechnikov";kwargs...)
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
    cost_function2 = function (p)
        ff = (t,u,du) -> prob.f(t,u,p,du)
        err = 0
        #du = zeros(length(tpoints))
        for i in 1:n
            err += (ff(tpoints[i],estimated_solution[i],estimated_derivative[i]) - estimated_derivative[i])^2
        end
        return err
    end
end


# cost_function = two_stage_method(prob,tpoints,data)
# result = optimize(cost_function, -20.0, 20.0)
# approximate_estimate = result.minimizer[1]

