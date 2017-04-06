@compat abstract type DECostFunction end

type CostVData{T,D,L} <: DECostFunction
  t::T
  data::D
  loss_func::L
end

function (f::CostVData)(sol)
  fill_length = length(f.t)-length(sol)
  for i in 1:fill_length
    push!(sol.u,fill(NaN,size(sol[1])))
  end
  y = vec(vecvec_to_mat(sol.u))
  norm(value(f.loss_func(),vec(f.data),vec(y)))
end

CostVData(t,data;loss_func = L2DistLoss) = CostVData(t,data,loss_func)
