export L2Loss, Regularization, LogLikeLoss, prior_loss, l2lossgradient!,
    colloc_grad

"""
    Regularization(λ, penalty = L2Penalty())

A regularization term for use with an objective builder such as
[`build_loss_objective`](@ref).

Calling a `Regularization` on a parameter vector `p` returns
`λ * value(penalty, p)`, i.e. the penalty evaluated on `p` scaled by `λ`.
The penalty is any penalty function from
[PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl); when
omitted it defaults to `L2Penalty()`, giving standard L2 (ridge) regularization.

# Fields

  - `λ`: the scalar weight applied to the penalty term.
  - `penalty`: the penalty function evaluated on the parameter vector.
"""
struct Regularization{L, P} <: DiffEqBase.DECostFunction
    λ::L
    penalty::P
end
Regularization(λ) = Regularization{typeof(λ), typeof(L2Penalty())}(λ, L2Penalty())

function (f::Regularization)(p)
    return f.λ * value(f.penalty, p)
end

"""
    prior_loss(prior, p)

Return the negative log prior of the parameter vector `p` under `prior`.

If `eltype(prior) <: UnivariateDistribution`, `prior` is treated as a collection
of univariate distributions and the result is `-sum(logpdf(prior[i], p[i]))`.
Otherwise `prior` is treated as a single (multivariate) distribution and the
result is `-logpdf(prior, p)`. Adding this term to a loss turns a maximum
likelihood objective into a maximum a posteriori (MAP) objective; it is used
internally by [`build_loss_objective`](@ref) and
[`multiple_shooting_objective`](@ref) when `priors` is supplied.
"""
function prior_loss(prior, p)
    ll = 0.0
    if eltype(prior) <: UnivariateDistribution
        for i in 1:length(prior)
            ll -= logpdf(prior[i], p[i])
        end
    else
        ll -= logpdf(prior, p)
    end
    return ll
end

"""
    L2Loss(t, data; differ_weight = nothing, data_weight = nothing,
        colloc_grad = nothing, dudt = nothing)

An optimized L2-distance loss for fitting a differential equation solution to
data. Calling an `L2Loss` on a solution `sol` returns the (weighted) sum of
squared residuals between `sol` and `data` at the timepoints `t`, returning
`Inf` if the solve was unsuccessful.

# Arguments

  - `t`: the timepoints at which the data are given.
  - `data`: the measured values, where column `i` holds the state at `t[i]`. A
    vector is reshaped to a `1 x N` matrix.

# Keyword Arguments

  - `data_weight`: a scalar or array of weights matching the size of `data`, used
    to weight each squared residual. `nothing` (the default) gives uniform unit
    weights. Minimizing a weighted `L2Loss` is equivalent to maximum likelihood
    estimation of a heteroskedastic Normal likelihood.
  - `differ_weight`: a scalar or array weight on the first-difference residuals
    `sol[i] - sol[i-1]` against the data first differences, which smooths the
    loss and can improve identifiability (e.g. for stochastic models). `nothing`
    (the default) disables the first-difference term.
  - `colloc_grad`: a matrix of collocation gradients for the data (see
    [`colloc_grad`](@ref)). When supplied, an interpolation-derivative term is
    added to the loss; combined with regularization this makes the loss
    equivalent to a 4DVAR objective.
  - `dudt`: a buffer used to accumulate the derivative estimates when
    `colloc_grad` is used; allocated automatically when `colloc_grad` is given.

# Fields

  - `t`, `data`, `differ_weight`, `data_weight`, `colloc_grad`, `dudt`: as above.
  - `du_buf`: an internal buffer for the derivative evaluation used with
    `colloc_grad`.
"""
struct L2Loss{T, D, U, W, G, B} <: DiffEqBase.DECostFunction
    t::T
    data::D
    differ_weight::U
    data_weight::W
    colloc_grad::G
    dudt::G
    du_buf::B
end

function (f::L2Loss)(sol::SciMLBase.AbstractNoTimeSolution)
    data = f.data
    weight = f.data_weight
    diff_weight = f.differ_weight
    colloc_grad = f.colloc_grad
    dudt = f.dudt

    if sol isa SciMLBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol.u)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf

    sumsq = 0.0

    solu = sol.u
    if weight === nothing
        @inbounds for i in 1:length(solu)
            sumsq += (data[i] - solu[i])^2
        end
    else
        @inbounds for i in 1:length(solu)
            if weight isa Real
                sumsq = sumsq + ((data[i] - solu[i])^2) * weight
            else
                sumsq = sumsq + ((data[i] - solu[i])^2) * weight[i]
            end
        end
    end
    return sumsq
end

function (f::L2Loss)(sol::SciMLBase.AbstractSciMLSolution)
    data = f.data
    weight = f.data_weight
    diff_weight = f.differ_weight
    colloc_grad = f.colloc_grad
    dudt = f.dudt

    if sol isa SciMLBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol.u)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf

    sumsq = 0.0

    nsteps = length(sol.u)
    if weight === nothing
        @inbounds for i in 1:nsteps
            for j in 1:length(sol.u[i])
                sumsq += (data[j, i] - sol[j, i])^2
            end
            if diff_weight !== nothing && i != 1
                for j in 1:length(sol.u[i])
                    if diff_weight isa Real
                        sumsq += diff_weight *
                            (
                            (
                                data[j, i] - data[j, i - 1] - sol[j, i] +
                                    sol[j, i - 1]
                            )^2
                        )
                    else
                        sumsq += diff_weight[j, i] *
                            (
                            (
                                data[j, i] - data[j, i - 1] - sol[j, i] +
                                    sol[j, i - 1]
                            )^2
                        )
                    end
                end
            end
        end
    else
        @inbounds for i in 1:nsteps
            if weight isa Real
                for j in 1:length(sol.u[i])
                    sumsq = sumsq + ((data[j, i] - sol[j, i])^2) * weight
                end
            else
                for j in 1:length(sol.u[i])
                    sumsq = sumsq + ((data[j, i] - sol[j, i])^2) * weight[j, i]
                end
            end
            if diff_weight !== nothing && i != 1
                for j in 1:length(sol.u[i])
                    if diff_weight isa Real
                        sumsq += diff_weight *
                            (
                            (
                                data[j, i] - data[j, i - 1] - sol[j, i] +
                                    sol[j, i - 1]
                            )^2
                        )
                    else
                        sumsq += diff_weight[j, i] *
                            (
                            (
                                data[j, i] - data[j, i - 1] - sol[j, i] +
                                    sol[j, i - 1]
                            )^2
                        )
                    end
                end
            end
        end
    end
    if colloc_grad !== nothing
        du_buf = f.du_buf
        for i in 1:size(colloc_grad)[2]
            sol.prob.f.f(du_buf, sol.u[i], sol.prob.p, sol.t[i])
            dudt[:, i] .= du_buf
        end
        sumsq += sum(abs2, x - y for (x, y) in zip(dudt, colloc_grad))
    end
    return sumsq
end

# Cost functions are written assuming a data matrix
# Turn vectors into a 1xN matrix
matrixize(x) = x isa Vector ? reshape(x, 1, length(x)) : x

function L2Loss(
        t, data; differ_weight = nothing, data_weight = nothing,
        colloc_grad = nothing,
        dudt = nothing
    )
    return L2Loss(
        t, matrixize(data), matrixize(differ_weight),
        matrixize(data_weight), matrixize(colloc_grad),
        colloc_grad === nothing ? nothing : zeros(size(colloc_grad)),
        colloc_grad === nothing ? nothing : zeros(size(colloc_grad, 1))
    )
end

function (f::L2Loss)(sol::SciMLBase.AbstractEnsembleSolution)
    return mean(f.(sol.u))
end

"""
    colloc_grad(t, data)

Estimate the time-derivative of `data` by spline collocation, for use as the
`colloc_grad` argument of [`L2Loss`](@ref).

For each state (row of `data`), a cubic (3rd order) `Dierckx.Spline1D` is fit to
`data` against `t` and differentiated at the timepoints `t`. The per-timepoint
derivatives are collected into a matrix of the same shape as `data`, where
column `i` is the estimated derivative at `t[i]`.

# Arguments

  - `t`: the timepoints, a length-`N` vector.
  - `data`: an `m x N` matrix of measured state values.

# Returns

An `m x N` matrix of the collocation-estimated derivatives.
"""
function colloc_grad(t::T, data::D) where {T, D}
    splines = [Dierckx.Spline1D(t, data[i, :]) for i in 1:size(data)[1]]
    grad = [Dierckx.derivative(spline, t[1:end]) for spline in splines]
    grad = [[grad[1][i], grad[2][i]] for i in 1:length(grad[1])]
    grad = convert(Array, VectorOfArray(grad))
    return grad
end

"""
    LogLikeLoss(t, data_distributions)
    LogLikeLoss(t, data_distributions, diff_distributions)

A negative log-likelihood loss for fitting a differential equation solution to a
field of distributions. Calling a `LogLikeLoss` on a solution `sol` returns the
negative total log-likelihood of `sol` under `data_distributions` at the
timepoints `t` (so minimizing it performs maximum likelihood estimation),
returning `Inf` if the solve was unsuccessful.

There are two forms for `data_distributions`:

  - If `data_distributions[i, j]` is a `UnivariateDistribution`, it gives the
    likelihood at `t[i]` for component `j`.
  - If `data_distributions[i]` is a `MultivariateDistribution`, it gives the
    likelihood at `t[i]` over the full state vector.

These distributions can be produced with `fit_mle` on a dataset against a chosen
distribution type.

# Arguments

  - `t`: the timepoints at which the distributions apply.
  - `data_distributions`: the field of likelihood distributions (a vector is
    reshaped to a `1 x N` matrix).
  - `diff_distributions`: an optional field of distributions placed on the
    first-difference terms `sol[i] - sol[i-1]`, contributing an additional
    log-likelihood term. When supplied via the three-argument constructor its
    contribution is scaled by `weight = 1`.

# Fields

  - `t`, `data_distributions`, `diff_distributions`: as above.
  - `weight`: the scalar weight applied to the first-difference log-likelihood
    term (`nothing` when `diff_distributions` is not used).
"""
struct LogLikeLoss{T, D} <: DiffEqBase.DECostFunction
    t::T
    data_distributions::D
    diff_distributions::L where {L <: Union{Nothing, D}}
    weight::Any
end
function LogLikeLoss(t, data_distributions)
    return LogLikeLoss(t, matrixize(data_distributions), nothing, nothing)
end
function LogLikeLoss(t, data_distributions, diff_distributions)
    return LogLikeLoss(t, matrixize(data_distributions), matrixize(diff_distributions), 1)
end

function (f::LogLikeLoss)(sol::SciMLBase.AbstractSciMLSolution)
    distributions = f.data_distributions
    if sol isa SciMLBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol.u)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf
    ll = 0.0

    if eltype(distributions) <: UnivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol.u[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            ll -= logpdf(distributions[i, j], sol[i, j])
        end
    else # MultivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol.u[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            ll -= logpdf(distributions[i], sol.u[i])
        end
    end

    if f.diff_distributions !== nothing
        distributions = f.diff_distributions
        diff_data = sol.u[2:end] - sol.u[1:(end - 1)]
        fill_length = length(f.t) - length(diff_data)
        for i in 1:fill_length
            push!(diff_data, fill(Inf, size(sol.u[1])))
        end
        fdll = 0
        if eltype(distributions) <: UnivariateDistribution
            for j in 1:(length(f.t) - 1), i in 1:length(diff_data[1])

                fdll -= logpdf(distributions[j, i], diff_data[j][i])
            end
        else
            for j in 1:(length(f.t) - 1)
                fdll -= logpdf(distributions[j], diff_data[j])
            end
        end
        ll += f.weight * fdll
    end

    return ll
end

function (f::LogLikeLoss)(sol::SciMLBase.AbstractEnsembleSolution)
    distributions = f.data_distributions
    failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol.u)
    failure && return Inf
    ll = 0.0
    if eltype(distributions) <: UnivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol.u[1].u[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            vals = [s[i, j] for s in sol.u]
            ll -= loglikelihood(distributions[i, j], vals)
        end
    else
        for j in 1:length(f.t)
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            vals = [s[i, j] for i in 1:length(sol.u[1].u[1]), s in sol.u]
            ll -= loglikelihood(distributions[j], vals)
        end
    end

    if f.diff_distributions !== nothing
        distributions = f.diff_distributions
        fdll = 0
        if eltype(distributions) <: UnivariateDistribution
            for j in 2:length(f.t), i in 1:length(sol.u[1].u[1])

                vals = [s[i, j] - s[i, j - 1] for s in sol.u]
                fdll -= logpdf(distributions[j - 1, i], vals)[1]
            end
        else
            for j in 2:length(f.t)
                vals = [s[i, j] - s[i, j - 1] for i in 1:length(sol.u[1].u[1]), s in sol.u]
                fdll -= logpdf(distributions[j - 1], vals)[1]
            end
        end
        ll += f.weight * fdll
    end

    return ll
end

"""
    l2lossgradient!(grad, sol, data, sensitivities, num_p)

Compute, in place, the gradient of an L2 loss with respect to `num_p`
parameters and write it into `grad`.

Given a solution `sol`, the target `data`, and the parameter `sensitivities`
(where `sensitivities[i]` is the derivative of the solution with respect to
parameter `i`), this accumulates
`grad[i] -= sum(2 * (data - sol) .* sensitivities[i])` over all state components
and timepoints. `grad` is zeroed before accumulation. Returns `nothing`.

# Arguments

  - `grad`: a length-`num_p` vector, overwritten with the loss gradient.
  - `sol`: the solution values, shaped like `data`.
  - `data`: the target data.
  - `sensitivities`: a collection of length `num_p` of parameter sensitivities,
    each shaped like `data`.
  - `num_p`: the number of parameters.
"""
function l2lossgradient!(grad, sol, data, sensitivities, num_p)
    fill!(grad, 0.0)
    data_x_size = size(data, 1)
    my_grad = @. 2 * (data - sol)
    u0len = length(data[1])
    K = size(my_grad, 2)
    for k in 1:K, i in 1:num_p, j in 1:data_x_size
        grad[i] -= my_grad[j, k] * sensitivities[i][j, k]
    end
    return
end
