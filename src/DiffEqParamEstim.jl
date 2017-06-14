__precompile__()

module DiffEqParamEstim
using DiffEqBase, LsqFit, LossFunctions, RecursiveArrayTools, ForwardDiff, Calculus

using Compat

include("cost_functions.jl")
include("lm_fit.jl")
include("build_loss_objective.jl")
include("build_lsoptim_objective.jl")
include("build_maximum_likelihood_objective.jl")
include("two_stage_method.jl")
include("kernels.jl")

end # module
