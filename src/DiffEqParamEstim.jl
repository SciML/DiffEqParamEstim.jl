__precompile__()

module DiffEqParamEstim
using DiffEqBase, LsqFit, LossFunctions, RecursiveArrayTools, ForwardDiff, Calculus, Optim


include("lm_fit.jl")
include("build_loss_objective.jl")
include("build_lsoptim_objective.jl")
include("two_stage_method.jl")



end # module
