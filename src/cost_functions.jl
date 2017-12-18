export DECostFunction, CostVData, L2Loss, Regularization, LogLikelihood

struct Regularization{L,P} <: DECostFunction
  λ::L
  penalty::P
end
Regularization(λ) = Regularization{typeof(λ),typeof(L2Penalty())}(λ,L2Penalty())

function (f::Regularization)(p)
  f.λ*value(f.penalty, p)
end

struct LogLikelihood{T,D1,D2,C} <: DECostFunction
  t::T
  data::D1
  distribution::D2
  covariance::C
end
function (f::LogLikelihood)(sol::DESolution)
  fill_length = length(f.t)-length(sol)
  covariance = f.covariance
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  likelihood_cost = 0.0
  if covariance == nothing
    @inbounds for i in 1:length(sol)
      for j in 1:length(sol[i])
        likelihood_cost +=loglikelihood(f.distribution(sol[j,i],1),f.data[j,i])
      end
    end
  else
    @inbounds for i in 1:length(sol)
      for j in 1:length(sol[i])
        likelihood_cost +=loglikelihood(f.distribution(sol[j,i],covariance[j,i]),f.data[j,i])
      end
    end
  end
  likelihood_cost
end

LogLikelihood(t,data;distribution = Normal,covariance=nothing) = LogLikelihood(t,data,distribution,covariance)

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
