export DECostFunction, CostVData, L2Loss, Regularization, LogLikeLoss

struct Regularization{L,P} <: DECostFunction
  λ::L
  penalty::P
end
Regularization(λ) = Regularization{typeof(λ),typeof(L2Penalty())}(λ,L2Penalty())

function (f::Regularization)(p)
  f.λ*value(f.penalty, p)
end

struct CostVData{T,D,L,W} <: DECostFunction
  t::T
  data::D
  loss_func::L
  weight::W
end

function (f::CostVData)(sol::DESolution)
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  if f.weight == nothing
    norm(value(f.loss_func(),vec(f.data),vec(sol)))
  else
    norm(value(f.loss_func(),vec(f.data),vec(sol)).*vec(f.weight))
  end
end

function (f::CostVData)(sol::AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

CostVData(t,data;loss_func = L2DistLoss,weight=nothing) = CostVData(t,data,loss_func,weight)

struct L2Loss{T,D,W} <: DECostFunction
  t::T
  data::D
  weight::W
end

function (f::L2Loss)(sol::DESolution)
  data = f.data
  weight = f.weight
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  sumsq = 0.0
  if weight == nothing
    @inbounds for i in 1:length(sol)
      for j in 1:length(sol[i])
        sumsq +=(data[j,i] - sol[j,i])^2
      end
    end
  else
    @inbounds for i in 1:length(sol)
      for j in 1:length(sol[i])
        sumsq = sumsq + ((data[j,i] - sol[j,i])^2)*weight[j,i]
      end
    end
  end
  sumsq
end
L2Loss(t,data;weight=nothing) = L2Loss(t,data,weight)

function (f::L2Loss)(sol::AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

struct LogLikeLoss{T,D} <: DECostFunction
  t::T
  distributions::D
end

function (f::LogLikeLoss)(sol::DESolution)
  distributions = f.distributions
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
      ll -= logpdf(f.distributions[i,j],sol[i,j])
    end
  else # MultivariateDistribution
    for j in 1:length(f.t), i in 1:length(sol[1][1])
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      ll -= logpdf(f.distributions[i],sol[i])
    end
  end
  ll
end

function (f::LogLikeLoss)(sol::AbstractMonteCarloSolution)
  distributions = f.distributions
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
      vals = [s[i,j] for s in sol]
      ll -= loglikelihood(f.distributions[i,j],vals)
    end
  else
    for j in 1:length(f.t)
      # i is the number of time points
      # j is the size of the system
      # corresponds to distributions[i,j]
      vals = [s[i,j] for i in length(sol[1][1]) for s in sol]
      ll -= loglikelihood(f.distributions[i],vals)
    end
  end
  ll
end
