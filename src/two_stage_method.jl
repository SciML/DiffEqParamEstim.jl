export TwoStageCost, two_stage_method
export EpanechnikovKernel, UniformKernel, TriangularKernel, QuarticKernel
export TriweightKernel, TricubeKernel, GaussianKernel, CosineKernel
export LogisticKernel, SigmoidKernel, SilvermanKernel

struct TwoStageCost{F, F2, D} <: Function
    cost_function::F
    cost_function2::F2
    estimated_solution::D
    estimated_derivative::D
end

(f::TwoStageCost)(p) = f.cost_function(p)
(f::TwoStageCost)(p, g) = f.cost_function2(p, g)

function decide_kernel(kernel::Symbol)
    if kernel == :Epanechnikov
        return EpanechnikovKernel()
    elseif kernel == :Uniform
        return UniformKernel()
    elseif kernel == :Triangular
        return TriangularKernel()
    elseif kernel == :Quartic
        return QuarticKernel()
    elseif kernel == :Triweight
        return TriweightKernel()
    elseif kernel == :Tricube
        return TricubeKernel()
    elseif kernel == :Gaussian
        return GaussianKernel()
    elseif kernel == :Cosine
        return CosineKernel()
    elseif kernel == :Logistic
        return LogisticKernel()
    elseif kernel == :Sigmoid
        return SigmoidKernel()
    elseif kernel == :Silverman
        return SilvermanKernel()
    else
        return error("Kernel name not recognized")
    end
end

function construct_iip_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    function (p)
        _du = PreallocationTools.get_tmp(du, p)
        vecdu = vec(_du)
        cost = zero(first(p))
        for i in eachindex(preview_est_sol)
            est_sol = preview_est_sol[i]
            f(_du, est_sol, p, tpoints[i])
            vecdu .= vec(preview_est_deriv[i]) .- vec(_du)
            cost += sum(abs2, vecdu)
        end
        cost
    end
end

function construct_oop_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    function (p)
        cost = zero(first(p))
        for i in eachindex(preview_est_sol)
            est_sol = preview_est_sol[i]
            _du = f(est_sol, p, tpoints[i])
            cost += sum(abs2, vec(preview_est_deriv[i]) .- vec(_du))
        end
        cost
    end
end

get_chunksize(cs) = cs
get_chunksize(cs::Type{Val{CS}}) where {CS} = CS

function two_stage_method(prob::DiffEqBase.DEProblem, tpoints, data;
                          kernel::Union{CollocationKernel, Symbol} = EpanechnikovKernel(),
                          loss_func = L2Loss, mpg_autodiff = false,
                          verbose = false, verbose_steps = 100,
                          autodiff_chunk = length(prob.p))
    if kernel isa Symbol
        @warn "Passing kernels as Symbols will be deprecated"
        kernel = decide_kernel(kernel)
    end

    # Step - 1

    estimated_derivative, estimated_solution = collocate_data(data, tpoints, kernel)

    # Step - 2

    du = PreallocationTools.dualcache(similar(prob.u0), autodiff_chunk)
    preview_est_sol = [@view estimated_solution[:, i]
                       for i in axes(estimated_solution, 2)]
    preview_est_deriv = [@view estimated_derivative[:, i]
                         for i in axes(estimated_solution, 2)]
    f = prob.f
    if DiffEqBase.isinplace(prob)
        cost_function = construct_iip_cost_function(f, du, preview_est_sol,
                                                    preview_est_deriv, tpoints)
    else
        cost_function = construct_oop_cost_function(f, du, preview_est_sol,
                                                    preview_est_deriv, tpoints)
    end

    if mpg_autodiff
        gcfg = ForwardDiff.GradientConfig(cost_function, prob.p,
                                          ForwardDiff.Chunk{get_chunksize(autodiff_chunk)}())
        g! = (x, out) -> ForwardDiff.gradient!(out, cost_function, x, gcfg)
    else
        g! = (x, out) -> Calculus.finite_difference!(cost_function, x, out, :central)
    end
    if verbose
        count = 0 # keep track of # function evaluations
    end
    cost_function2 = function (p, grad)
        if length(grad) > 0
            g!(p, grad)
        end
        loss_val = cost_function(p)
        if verbose
            count::Int += 1
            if mod(count, verbose_steps) == 0
                println("Iteration: $count")
                println("Current Cost: $loss_val")
                println("Parameters: $p")
            end
        end
        loss_val
    end

    return TwoStageCost(cost_function, cost_function2, estimated_solution,
                        estimated_derivative)
end
