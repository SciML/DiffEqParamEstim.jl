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

struct L2Loss{T,D,U,W} <: DECostFunction
  t::T
  data::D
  differ_weight::U
  data_weight::W
end

function (f::L2Loss)(sol::DESolution)
  data = f.data
  weight = f.data_weight
  diff_weight = f.differ_weight
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
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
      for j in 1:length(sol[i])
        sumsq = sumsq + ((data[j,i] - sol[j,i])^2)*weight[j,i]
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
L2Loss(t,data) = L2Loss(t,data,nothing,nothing)
L2Loss(t,data;data_weight=1.0,differ_weight=nothing) = L2Loss(t,data,data_weight,differ_weight)

function (f::L2Loss)(sol::AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

struct LogLikeLoss{T,D} <: DECostFunction
  t::T
  data_distributions::D
  diff_distributions::L where L<:Union{Void,D}
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

function (f::LogLikeLoss)(sol::AbstractMonteCarloSolution)
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
