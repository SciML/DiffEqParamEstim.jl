export L2Loss, Regularization, LogLikeLoss, prior_loss, l2lossgradient!,
       colloc_grad

struct Regularization{L, P} <: DiffEqBase.DECostFunction
    λ::L
    penalty::P
end
Regularization(λ) = Regularization{typeof(λ), typeof(L2Penalty())}(λ, L2Penalty())

function (f::Regularization)(p)
    f.λ * value(f.penalty, p)
end

function prior_loss(prior, p)
    ll = 0.0
    if eltype(prior) <: UnivariateDistribution
        for i in 1:length(prior)
            ll -= logpdf(prior[i], p[i])
        end
    else
        ll -= logpdf(prior, p)
    end
    ll
end

struct L2Loss{T, D, U, W, G} <: DiffEqBase.DECostFunction
    t::T
    data::D
    differ_weight::U
    data_weight::W
    colloc_grad::G
    dudt::G
end

function (f::L2Loss)(sol::DiffEqBase.AbstractNoTimeSolution)
    data = f.data
    weight = f.data_weight
    diff_weight = f.differ_weight
    colloc_grad = f.colloc_grad
    dudt = f.dudt

    if sol isa DiffEqBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf

    sumsq = 0.0

    if weight == nothing
        @inbounds for i in 1:length(sol)
            sumsq += (data[i] - sol.u[i])^2
        end
    else
        @inbounds for i in 1:length(sol)
            if weight isa Real
                sumsq = sumsq + ((data[i] - sol.u[i])^2) * weight
            else
                sumsq = sumsq + ((data[i] - sol.u[i])^2) * weight[i]
            end
        end
    end
    sumsq
end

function (f::L2Loss)(sol::SciMLBase.AbstractSciMLSolution)
    data = f.data
    weight = f.data_weight
    diff_weight = f.differ_weight
    colloc_grad = f.colloc_grad
    dudt = f.dudt

    if sol isa DiffEqBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf

    sumsq = 0.0

    if weight == nothing
        @inbounds for i in 1:length(sol)
            for j in 1:length(sol.u[i])
                sumsq += (data[j, i] - sol[j, i])^2
            end
            if diff_weight != nothing && i != 1
                for j in 1:length(sol.u[i])
                    if diff_weight isa Real
                        sumsq += diff_weight *
                                 ((data[j, i] - data[j, i - 1] - sol[j, i] +
                                   sol[j, i - 1])^2)
                    else
                        sumsq += diff_weight[j, i] *
                                 ((data[j, i] - data[j, i - 1] - sol[j, i] +
                                   sol[j, i - 1])^2)
                    end
                end
            end
        end
    else
        @inbounds for i in 1:length(sol)
            if weight isa Real
                for j in 1:length(sol.u[i])
                    sumsq = sumsq + ((data[j, i] - sol[j, i])^2) * weight
                end
            else
                for j in 1:length(sol.u[i])
                    sumsq = sumsq + ((data[j, i] - sol[j, i])^2) * weight[j, i]
                end
            end
            if diff_weight != nothing && i != 1
                for j in 1:length(sol.u[i])
                    if diff_weight isa Real
                        sumsq += diff_weight *
                                 ((data[j, i] - data[j, i - 1] - sol[j, i] +
                                   sol[j, i - 1])^2)
                    else
                        sumsq += diff_weight[j, i] *
                                 ((data[j, i] - data[j, i - 1] - sol[j, i] +
                                   sol[j, i - 1])^2)
                    end
                end
            end
        end
    end
    if colloc_grad != nothing
        for i in 1:size(colloc_grad)[2]
            sol.prob.f.f(@view(dudt[:, i]), sol.u[i], sol.prob.p, sol.t[i])
        end
        sumsq += sum(abs2, x - y for (x, y) in zip(dudt, colloc_grad))
    end
    sumsq
end

# Cost functions are written assuming a data matrix
# Turn vectors into a 1xN matrix
matrixize(x) = x isa Vector ? reshape(x, 1, length(x)) : x

function L2Loss(t, data; differ_weight = nothing, data_weight = nothing,
        colloc_grad = nothing,
        dudt = nothing)
    L2Loss(t, matrixize(data), matrixize(differ_weight),
        matrixize(data_weight), matrixize(colloc_grad),
        colloc_grad == nothing ? nothing : zeros(size(colloc_grad)))
end

function (f::L2Loss)(sol::DiffEqBase.AbstractEnsembleSolution)
    mean(f.(sol.u))
end

#t - 1xN array, data - mxN array, returns mxN array
function colloc_grad(t::T, data::D) where {T, D}
    splines = [Dierckx.Spline1D(t, data[i, :]) for i in 1:size(data)[1]]
    grad = [Dierckx.derivative(spline, t[1:end]) for spline in splines]
    grad = [[grad[1][i], grad[2][i]] for i in 1:length(grad[1])]
    grad = convert(Array, VectorOfArray(grad))
    return grad
end

struct LogLikeLoss{T, D} <: DiffEqBase.DECostFunction
    t::T
    data_distributions::D
    diff_distributions::L where {L <: Union{Nothing, D}}
    weight::Any
end
function LogLikeLoss(t, data_distributions)
    LogLikeLoss(t, matrixize(data_distributions), nothing, nothing)
end
function LogLikeLoss(t, data_distributions, diff_distributions)
    LogLikeLoss(t, matrixize(data_distributions), matrixize(diff_distributions), 1)
end

function (f::LogLikeLoss)(sol::SciMLBase.AbstractSciMLSolution)
    distributions = f.data_distributions
    if sol isa DiffEqBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol)
    else
        failure = !SciMLBase.successful_retcode(sol.retcode)
    end
    failure && return Inf
    ll = 0.0

    if eltype(distributions) <: UnivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            ll -= logpdf(distributions[i, j], sol[i, j])
        end
    else # MultivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            ll -= logpdf(distributions[i], sol.u[i])
        end
    end

    if f.diff_distributions != nothing
        distributions = f.diff_distributions
        diff_data = sol.u[2:end] - sol.u[1:(end - 1)]
        fill_length = length(f.t) - length(diff_data)
        for i in 1:fill_length
            push!(diff_data, fill(Inf, size(sol[1])))
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

    ll
end

function (f::LogLikeLoss)(sol::DiffEqBase.AbstractEnsembleSolution)
    distributions = f.data_distributions
    if sol_tmp isa DiffEqBase.AbstractEnsembleSolution
        failure = any(!SciMLBase.successful_retcode(s.retcode) for s in sol_tmp)
    else
        failure = !SciMLBase.successful_retcode(sol_tmp.retcode)
    end
    failure && return Inf
    ll = 0.0
    if eltype(distributions) <: UnivariateDistribution
        for j in 1:length(f.t), i in 1:length(sol[1])
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            vals = [s[i, j] for s in sol]
            ll -= loglikelihood(distributions[i, j], vals)
        end
    else
        for j in 1:length(f.t)
            # i is the number of time points
            # j is the size of the system
            # corresponds to distributions[i,j]
            vals = [s[i, j] for i in 1:length(sol[1]), s in sol]
            ll -= loglikelihood(distributions[j], vals)
        end
    end

    if f.diff_distributions != nothing
        distributions = f.diff_distributions
        fdll = 0
        if eltype(distributions) <: UnivariateDistribution
            for j in 2:length(f.t), i in 1:length(sol[1][1])
                vals = [s[i, j] - s[i, j - 1] for s in sol]
                fdll -= logpdf(distributions[j - 1, i], vals)[1]
            end
        else
            for j in 2:length(f.t)
                vals = [s[i, j] - s[i, j - 1] for i in 1:length(sol[1]), s in sol]
                fdll -= logpdf(distributions[j - 1], vals)[1]
            end
        end
        ll += f.weight * fdll
    end

    ll
end

function l2lossgradient!(grad, sol, data, sensitivities, num_p)
    fill!(grad, 0.0)
    data_x_size = size(data, 1)
    my_grad = @. 2 * (data - sol)
    u0len = length(data[1])
    K = size(my_grad, 2)
    for k in 1:K, i in 1:num_p, j in 1:data_x_size
        grad[i] -= my_grad[j, k] * sensitivities[i][j, k]
    end
end
