export L2Loss, Regularization, LogLikeLoss, prior_loss

struct Regularization{L,P} <: DiffEqBase.DECostFunction
  λ::L
  penalty::P
end
Regularization(λ) = Regularization{typeof(λ),typeof(L2Penalty())}(λ,L2Penalty())

function (f::Regularization)(p)
  f.λ*value(f.penalty, p)
end

function prior_loss(prior,p)
  ll = 0.0
  if eltype(prior) <: UnivariateDistribution
    for i in 1:length(prior)
      ll -= logpdf(prior[i],p[i])
    end
  else
    ll -= logpdf(prior,p)
  end
  ll
end

struct L2Loss{T,D,U,W} <: DiffEqBase.DECostFunction
  t::T
  data::D
  differ_weight::U
  data_weight::W
end

function (f::L2Loss)(sol::DiffEqBase.DESolution)
  data = f.data
  weight = f.data_weight
  diff_weight = f.differ_weight
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    if eltype(sol.u) <: Number
      push!(sol.u,Inf)
    else
      push!(sol.u,fill(Inf,size(sol[1])))
    end
  end
  sumsq = 0.0

  if weight == nothing
    @inbounds for i in 2:length(sol)
      for j in 1:length(sol[i])
        sumsq +=(data[j,i] - sol[j,i])^2
      end
      if diff_weight != nothing
          for j in 1:length(sol[i])
            if typeof(diff_weight) <: Real
              sumsq += diff_weight*((data[j,i] - data[j,i-1] - sol[j,i] + sol[j,i-1])^2)
            else
             sumsq += diff_weight[j,i]*((data[j,i] - data[j,i-1] - sol[j,i] + sol[j,i-1])^2)
            end
          end
      end
    end
  else
    @inbounds for i in 2:length(sol)
      if typeof(weight) <: Real
        for j in 1:length(sol[i])
          sumsq = sumsq + ((data[j,i] - sol[j,i])^2)*weight
        end
      else
        for j in 1:length(sol[i])
          sumsq = sumsq + ((data[j,i] - sol[j,i])^2)*weight[j,i]
        end
      end
      if diff_weight != nothing
        for j in 1:length(sol[i])
          if typeof(diff_weight) <: Real
            sumsq += diff_weight*((data[j,i] - data[j,i-1] - sol[j,i] + sol[j,i-1])^2)
          else
            sumsq += diff_weight[j,i]*((data[j,i] - data[j,i-1] - sol[j,i] + sol[j,i-1])^2)
          end
        end
      end
    end
  end
  sumsq
end
L2Loss(t,data;differ_weight=nothing,data_weight=nothing) = L2Loss(t,data,differ_weight,data_weight)

function (f::L2Loss)(sol::DiffEqBase.AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

struct LogLikeLoss{T,D} <: DiffEqBase.DECostFunction
  t::T
  data_distributions::D
  diff_distributions::L where L<:Union{Nothing,D}
  weight
end
LogLikeLoss(t,data_distributions) = LogLikeLoss(t,data_distributions,nothing,nothing)
LogLikeLoss(t,data_distributions,diff_distributions) = LogLikeLoss(t,data_distributions,diff_distributions,1)

function (f::LogLikeLoss)(sol::DESolution)
  distributions = f.data_distributions
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  ll = 0.0

  if eltype(distributions) <: UnivariateDistribution
    for j in 1:length(f.t), i in 1:length(sol[1][1])
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      ll -= logpdf(distributions[i,j],sol[i,j])
    end
  else # MultivariateDistribution
    for j in 1:length(f.t), i in 1:length(sol[1][1])
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      ll -= logpdf(distributions[i],sol[i])
    end
  end

  if f.diff_distributions != nothing
    distributions = f.diff_distributions
    diff_data = sol.u[2:end] - sol.u[1:end-1]
    fill_length = length(f.t)-length(diff_data)
    for i in 1:fill_length
      push!(diff_data,fill(Inf,size(sol[1])))
    end
    fdll = 0
    if eltype(distributions) <: UnivariateDistribution
      for j in 1:length(f.t)-1, i in 1:length(diff_data[1])
        fdll -= logpdf(distributions[j,i],diff_data[j][i])
      end
    else
      for j in 1:length(f.t)-1
        fdll -= logpdf(distributions[j],diff_data[j])
      end
    end
    ll += f.weight*fdll
  end

  ll
end

function (f::LogLikeLoss)(sol::DiffEqBase.AbstractMonteCarloSolution)
  distributions = f.data_distributions
  for s in sol
    fill_length = length(f.t)-length(s)
    for i in 1:fill_length
      push!(s.u,fill(Inf,size(s[1])))
    end
  end
  ll = 0.0
  if eltype(distributions) <: UnivariateDistribution
    for j in 1:length(f.t), i in 1:length(sol[1][1])
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      vals = [s[i,j] for s in sol]
      ll -= loglikelihood(distributions[i,j],vals)
    end
  else
    for j in 1:length(f.t)
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      vals = [s[i,j] for i in 1:length(sol[1][1]), s in sol]
      ll -= loglikelihood(distributions[j],vals)
    end
  end

  if f.diff_distributions != nothing
    distributions = f.diff_distributions
    fdll = 0
    if eltype(distributions) <: UnivariateDistribution
      for j in 2:length(f.t), i in 1:length(sol[1][1])
        vals = [s[i,j] - s[i,j-1] for s in sol]
        fdll -= logpdf(distributions[j-1,i],vals)[1]
      end
    else
      for j in 2:length(f.t)
        vals = [s[i,j] - s[i,j-1] for i in 1:length(sol[1][1]), s in sol]
        fdll -= logpdf(distributions[j-1],vals)[1]
      end
    end
    ll += f.weight*fdll
  end

  ll
end
