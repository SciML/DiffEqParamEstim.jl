export build_loss_objective

"""
    build_loss_objective(prob::SciMLBase.AbstractSciMLProblem, alg, loss,
        adtype = SciMLBase.NoAD(), regularization = nothing, args...;
        priors = nothing, prob_generator = STANDARD_PROB_GENERATOR, kwargs...)

Build an `OptimizationFunction` that solves `prob` for a candidate parameter
vector and reduces the resulting solution to a scalar via `loss`.

The returned function has the signature `cost(p, _ = nothing)`. For each `p` it
regenerates a problem with `prob_generator(prob, p)`, solves it with `alg`, and
returns `loss(sol)`. When `loss` is an `L2Loss` or `LogLikeLoss`, the solve
saves only at the loss timepoints (`saveat = loss.t`, `save_everystep = false`,
`dense = false`); otherwise the solve uses the passed keyword arguments as-is.

# Arguments

  - `prob`: the `AbstractSciMLProblem` (ODE, SDE, DAE, DDE, ...) to solve.
  - `alg`: the solver algorithm matching `prob`.
  - `loss`: a callable mapping a solution to a scalar cost, e.g. [`L2Loss`](@ref)
    or [`LogLikeLoss`](@ref), or any user-supplied `loss(sol)`.
  - `adtype`: the automatic differentiation choice passed to `OptimizationFunction`
    (defaults to `SciMLBase.NoAD()`).
  - `regularization`: an optional callable of `p` added to the loss, e.g.
    [`Regularization`](@ref). `nothing` (the default) adds no regularization.
  - `args...`: extra positional arguments forwarded to `solve`.

# Keyword Arguments

  - `priors`: an array of univariate distributions (one per parameter) or a
    multivariate distribution. When provided, the negative log prior computed by
    [`prior_loss`](@ref) is added to the loss, turning the objective into a MAP
    estimate.
  - `prob_generator`: a function `(prob, p) -> newprob` used to build the problem
    for each parameter vector. Defaults to `STANDARD_PROB_GENERATOR`, which
    `remake`s `prob` with autodiff-compatible element types and the new `p`.
  - `kwargs...`: extra keyword arguments forwarded to the differential equation
    `solve`.

# Returns

An `OptimizationFunction` wrapping the cost function with `adtype`, ready to be
used to construct an `OptimizationProblem`.
"""
function build_loss_objective(
        prob::SciMLBase.AbstractSciMLProblem, alg, loss,
        adtype = SciMLBase.NoAD(),
        regularization = nothing, args...;
        priors = nothing,
        prob_generator = STANDARD_PROB_GENERATOR,
        kwargs...
    )
    cost_function = function (p, _ = nothing)
        tmp_prob = prob_generator(prob, p)
        if loss isa Union{L2Loss, LogLikeLoss}
            sol = solve(
                tmp_prob, alg, args...; saveat = loss.t, save_everystep = false,
                dense = false, kwargs...
            )
        else
            sol = solve(tmp_prob, alg, args...; kwargs...)
        end

        loss_val = loss(sol)

        if priors !== nothing
            loss_val += prior_loss(priors, p)
        end
        if regularization !== nothing
            loss_val += regularization(p)
        end
        return loss_val
    end

    return OptimizationFunction(cost_function, adtype)
end
