module DiffEqParamEstim
using DiffEqBase, PenaltyFunctions,
      RecursiveArrayTools, Distributions,
      LinearAlgebra, Dierckx, SciMLBase

import PreallocationTools
function STANDARD_PROB_GENERATOR(prob, p)
    f = ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(SciMLBase.unwrapped_f(prob.f))
    remake(prob; u0 = eltype(p).(prob.u0), p = p, f = f)
end
function STANDARD_PROB_GENERATOR(prob::EnsembleProblem, p)
    f = ODEFunction{isinplace(prob.prob), SciMLBase.FullSpecialize}(SciMLBase.unwrapped_f(prob.prob.f))
    EnsembleProblem(remake(prob.prob; u0 = eltype(p).(prob.prob.u0), p = p, f = f),
        output_func = prob.output_func,
        prob_func = prob.prob_func,
        reduction = prob.reduction,
        u_init = prob.u_init)
end
STANDARD_MS_PROB_GENERATOR = function (prob, p, k)
    t0, tf = prob.tspan
    P, N = length(prob.p), length(prob.u0)
    K = Int((length(p) - P) / N)
    τ = range(t0, tf, length = K + 1)
    f = ODEFunction{isinplace(prob), SciMLBase.FullSpecialize}(SciMLBase.unwrapped_f(prob.f))
    remake(prob; u0 = p[(1 + (k - 1) * N):(k * N)], p = p[(end - P + 1):end],
        tspan = (τ[k], τ[k + 1]), f = f)
end

include("cost_functions.jl")
include("build_loss_objective.jl")
include("kernels.jl")
include("two_stage_method.jl")
include("multiple_shooting_objective.jl")

end # module
