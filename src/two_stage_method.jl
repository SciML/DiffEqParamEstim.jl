using Optim

export two_stage_method

# Step - 1
function cost_function1(b,t0,tpoints,data)
    err = 0
    for i in 1:length(tpoints)
        w_i = exp(tpoints[i]-t0)
        temp_f = b[1]-b[2]*(tpoints[i]-t0)
        err = err + w_i*(temp_f-data[i])^2
    end
    return err
end

# Step - 2
function cost_function4(p,tpoints,data,b0,b1,fff)
    err = 0
    #du = zeros(length(tpoints))
    for i in 1:length(tpoints)
        err = err + (b1[i] - fff(tpoints[i],b0[i]))^2
    end
    return err
end

function two_stage_method(prob::DEProblem,tpoints,data;kwargs...)
    #tpoints = [0.0,0.5,1.0]
    #data  = [1,exp(1),exp(2)]
    f = prob.f
    fff = (t,u) -> prob.f(t,u,p)
    b0 = []
    b1 = []
    for i in 1:length(tpoints)
        t0 = tpoints[i]
        result = optimize(b->cost_function1(b,t0,tpoints,data), [0.0,0.0])
        push!(b0,result.minimizer[1])
        push!(b1,result.minimizer[2])
    end

    return p->cost_function4(p,tpoints,data,b0,b1,fff)
end


# pf_func = function (t,u,p)
#     p*u
#  end

# pf = ParameterizedFunction(pf_func,[2])

# u0 = [1.0]
# tspan = (0.0,1.0)
# prob = ODEProblem(pf,u0,tspan)


# result = optimize(two_stage_method(prob,tpoints,data), 0.0, 20.0)
# result = optimize(p->cost_function3(p,tpoints,data,pf), 0.0, 20.0)
# result.minimizer[1]
