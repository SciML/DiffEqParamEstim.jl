export two_stage_method

# Step - 1
function cost_function1(b,t0,tpoints,data)
    err = 0
    for i in 1:length(tpoints)
        w_i = exp((tpoints[i]-t0)^2)
        temp_f = b[1]-(b[2]*(tpoints[i]-t0))
        err = err + w_i*(temp_f-data[i])^2
    end
    return err
end


function two_stage_method(prob::DEProblem,tpoints,data;kwargs...)
    f = prob.f
    b0 = []
    b1 = []
    for i in 1:length(tpoints)
        t0 = tpoints[i]
        result = optimize(b->cost_function1(b,t0,tpoints,data), [1.0,1.0])
        push!(b0,result.minimizer[1])
        push!(b1,result.minimizer[2])
    end

    
    # Step - 2
    cost_function2 = function (p)
        ff = (t,u) -> prob.f(t,u,p)
        err = 0
        #du = zeros(length(tpoints))
        for i in 1:length(tpoints)
            err = err + (ff(tpoints[i],b0[i]) - b1[i])^2
        end
        return err
    end
end

# using DifferentialEquations
# using Optim
# tpoints = [0.0,0.5,1.0]
# data  = [1,exp(1),exp(2)]

# pf_func = function (t,u,p)
#     p*u
#  end

# pf = ParameterizedFunction(pf_func,[2])

# u0 = [1.0]
# tspan = (0.0,1.0)
# prob = ODEProblem(pf,u0,tspan)

# cost_function = two_stage_method(prob,tpoints,data)
# result = optimize(cost_function, -20.0, 20.0)
# result.minimizer[1]

