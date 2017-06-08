using DiffEqParamEstim, Base.Test
include("test_examples.jl")

println("Use LM to fit the parameter")

fit = lm_fit(prob1,t,vec(data),[1.0],Tsit5(),show_trace=true,lambda=10000.0)
param = fit.param
@test param[1] ≈ 1.5 atol=1e-3

println("Multivariate")

fit = lm_fit(prob2,t,vec(data),[1.3,2.6],Tsit5(),show_trace=true,lambda=10000.0)
param = fit.param
@test param ≈ [1.5; 3.0] atol=0.002

fit = lm_fit(prob3,t,vec(data),[1.3,0.8,2.6,1.2],Tsit5(),show_trace=true,lambda=10000.0)
param = fit.param
@test param ≈ [1.5;1.0;3.0;1.0] atol=1e-2
