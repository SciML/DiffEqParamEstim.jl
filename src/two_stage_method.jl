export TwoStageCost, two_stage_objective

"""
    TwoStageCost(cost_function, estimated_solution, estimated_derivative)

The callable cost object produced by [`two_stage_objective`](@ref).

Calling a `TwoStageCost` on a parameter vector `p` forwards to
`cost_function(p, _p)`, which measures how well the differential equation's
right-hand side evaluated at the collocation-smoothed solution matches the
collocation-smoothed derivative. The smoothed trajectory and its derivative are
stored so they can be inspected after construction.

# Fields

  - `cost_function`: the underlying cost closure of signature `(p, _p = nothing)`.
  - `estimated_solution`: the kernel-smoothed estimate of the state trajectory at
    the collocation points.
  - `estimated_derivative`: the kernel-smoothed estimate of the state derivative
    at the collocation points.
"""
struct TwoStageCost{F, D} <: Function
    cost_function::F
    estimated_solution::D
    estimated_derivative::D
end

(f::TwoStageCost)(p, _p = nothing) = f.cost_function(p, _p)

decide_kernel(kernel::CollocationKernel) = kernel
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
    else
        return SilvermanKernel()
    end
end

function construct_t1(t, tpoints)
    return hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t)
end
function construct_t2(t, tpoints)
    return hcat(ones(eltype(tpoints), length(tpoints)), tpoints .- t, (tpoints .- t) .^ 2)
end
function construct_w(t, tpoints, h, kernel)
    W = @. calckernel((kernel,), (tpoints - t) / h) / h
    return Diagonal(W)
end
function construct_estimated_solution_and_derivative!(data, kernel, tpoints)
    _one = oneunit(first(data))
    _zero = zero(first(data))
    e1 = [_one; _zero]
    e2 = [_zero; _one; _zero]
    n = length(tpoints)
    h = (n^(-1 / 5)) * (n^(-3 / 35)) * ((log(n))^(-1 / 16))

    Wd = similar(data, n, size(data, 1))
    WT1 = similar(data, n, 2)
    WT2 = similar(data, n, 3)
    x = map(tpoints) do _t
        T1 = construct_t1(_t, tpoints)
        T2 = construct_t2(_t, tpoints)
        W = construct_w(_t, tpoints, h, kernel)
        mul!(Wd, W, data')
        mul!(WT1, W, T1)
        mul!(WT2, W, T2)
        (e2' * ((T2' * WT2) \ T2')) * Wd, (e1' * ((T1' * WT1) \ T1')) * Wd
    end
    estimated_derivative = reduce(hcat, transpose.(first.(x)))
    estimated_solution = reduce(hcat, transpose.(last.(x)))
    return estimated_derivative, estimated_solution
end

function construct_iip_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    return function (p, _ = nothing)
        _du = PreallocationTools.get_tmp(PreallocationTools.DiffCache(du, length(p)), p)
        vecdu = vec(_du)
        cost = zero(first(p))
        for i in 1:length(preview_est_sol)
            est_sol = preview_est_sol[i]
            f(_du, est_sol, p, tpoints[i])
            vecdu .= vec(preview_est_deriv[i]) .- vec(_du)
            cost += sum(abs2, vecdu)
        end
        return cost
    end
end

function construct_oop_cost_function(f, du, preview_est_sol, preview_est_deriv, tpoints)
    return function (p, _ = nothing)
        cost = zero(first(p))
        for i in 1:length(preview_est_sol)
            est_sol = preview_est_sol[i]
            _du = f(est_sol, p, tpoints[i])
            cost += sum(abs2, vec(preview_est_deriv[i]) .- vec(_du))
        end
        return cost
    end
end

"""
    two_stage_objective(prob::SciMLBase.AbstractDEProblem, tpoints, data,
        adtype = SciMLBase.NoAD(); kernel = EpanechnikovKernel())

Build an `OptimizationFunction` for the two-stage (non-parametric collocation)
parameter estimation method.

The two-stage method avoids repeatedly solving the differential equation: it
first builds a kernel-smoothed estimate of the state trajectory and its
derivative from `data`, then measures, for a candidate `p`, the squared
mismatch between the model right-hand side `prob.f` evaluated at the smoothed
state and the smoothed derivative. It is less accurate than solve-based
objectives but much faster, and is a good way to get into the neighborhood of
good parameters before refining with another method.

# Arguments

  - `prob`: the `AbstractDEProblem` whose right-hand side is being fit. Both
    in-place and out-of-place `prob.f` are supported.
  - `tpoints`: the timepoints at which `data` is sampled.
  - `data`: the measured state values used for the collocation smoothing.
  - `adtype`: the automatic differentiation choice passed to `OptimizationFunction`
    (defaults to `SciMLBase.NoAD()`).

# Keyword Arguments

  - `kernel`: the collocation smoothing kernel, either a `CollocationKernel`
    instance or a `Symbol` selecting one (`:Epanechnikov`, `:Uniform`,
    `:Triangular`, `:Quartic`, `:Triweight`, `:Tricube`, `:Gaussian`,
    `:Cosine`, `:Logistic`, `:Sigmoid`, `:Silverman`). Defaults to
    `EpanechnikovKernel()`.

# Returns

An `OptimizationFunction` wrapping a [`TwoStageCost`](@ref) with `adtype`. The
returned cost also exposes the smoothed `estimated_solution` and
`estimated_derivative`.
"""
function two_stage_objective(
        prob::SciMLBase.AbstractDEProblem, tpoints, data,
        adtype = SciMLBase.NoAD();
        kernel = EpanechnikovKernel()
    )
    f = prob.f
    kernel_function = decide_kernel(kernel)
    estimated_derivative,
        estimated_solution = construct_estimated_solution_and_derivative!(
        data,
        kernel_function,
        tpoints
    )

    # Step - 2
    preview_est_sol = [
        @view estimated_solution[:, i]
            for i in 1:size(estimated_solution, 2)
    ]
    preview_est_deriv = [
        @view estimated_derivative[:, i]
            for i in 1:size(estimated_solution, 2)
    ]

    cost_function = if isinplace(prob)
        construct_iip_cost_function(f, prob.u0, preview_est_sol, preview_est_deriv, tpoints)
    else
        construct_oop_cost_function(f, prob.u0, preview_est_sol, preview_est_deriv, tpoints)
    end

    return OptimizationFunction(
        TwoStageCost(
            cost_function, estimated_solution,
            estimated_derivative
        ), adtype
    )
end
