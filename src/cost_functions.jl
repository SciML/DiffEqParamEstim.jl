type CostVData{T,D,L} <: DECostFunction
  t::T
  data::D
  loss_func::L
end

function (f::CostVData)(sol::DESolution)
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  y = vec(vecvec_to_mat(sol.u))
  norm(value(f.loss_func(),vec(f.data),vec(y)))
end

function (f::CostVData)(sol::AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

CostVData(t,data;loss_func = L2DistLoss) = CostVData(t,data,loss_func)

type L2Loss{T,D} <: DECostFunction
  t::T
  data::D
end

function (f::L2Loss)(sol::DESolution)
  data = f.data
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  sumsq = 0.0
  @inbounds for i in 1:length(sol)
    for j in 1:length(sol[i])
      sumsq += (data[j,i] - sol[j,i])^2
    end
  end
  sumsq
end

function (f::L2Loss)(sol::AbstractMonteCarloSolution)
  mean(f.(sol.u))
end

type MaximumLikelihood{T,D,V} <: DECostFunction
  t::T
  data::D
  variance::V
end

function (f::MaximumLikelihood)(sol::DESolution)
  data = f.data
  variance = f.variance
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(Inf,size(sol[1])))
  end
  prod = 1.0
  @inbounds for i in 1:length(sol)
    for j in 1:length(sol[i])
      prod *= exp(((data[j,i] - sol[j,i])^2)/(-2*variance[j,i]))/(sqrt(2*π*variance[j,i]))
    end
  end
  prod
end
