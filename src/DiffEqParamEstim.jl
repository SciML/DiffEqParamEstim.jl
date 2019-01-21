module DiffEqParamEstim
using DiffEqBase, LsqFit, PenaltyFunctions,
      RecursiveArrayTools, ForwardDiff, Calculus, Distributions, LinearAlgebra, DiffEqSensitivity

STANDARD_PROB_GENERATOR(prob,p) = remake(prob;u0=eltype(p).(prob.u0),p=p)
STANDARD_PROB_GENERATOR(prob::MonteCarloProblem,p) = MonteCarloProblem(remake(prob.prob;u0=eltype(p).(prob.prob.u0),p=p))

include("cost_functions.jl")
include("lm_fit.jl")
include("build_loss_objective.jl")
include("build_lsoptim_objective.jl")
include("two_stage_method.jl")
include("kernels.jl")
include("multiple_shooting_objective.jl")

end # module
