using OrdinaryDiffEq, DiffEqParamEstim, Distributions, Test,RecursiveArrayTools
      # ,BlackBoxOptim,

pf_func = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3.0 * u[2] + u[1]*u[2]
end
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0]
prob1 = ODEProblem(pf_func,u0,tspan,p)
sol = solve(prob1,Tsit5())
t = collect(range(0, stop=10, length=200))
function generate_data(sol,t)
  randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
  data = convert(Array,randomized)
end
aggregate_data = convert(Array,VectorOfArray([generate_data(sol,t) for i in 1:100]))

distributions = [fit_mle(Normal,aggregate_data[i,j,:]) for i in 1:2, j in 1:200]
obj = build_loss_objective(prob1,Tsit5(),LogLikeLoss(t,distributions),
                                     maxiters=10000,verbose=false)
bound1 = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5)]
result = bboptimize(obj;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5,1.0] atol = 1e-1


data_distributions = [fit_mle(Normal,aggregate_data[i,j,:]) for i in 1:2, j in 1:200]
diff_distributions = [fit_mle(Normal,aggregate_data[i,j,:]-aggregate_data[i,j-1,:]) for j in 2:200, i in 1:2 ]
obj = build_loss_objective(prob1,Tsit5(),LogLikeLoss(t,data_distributions,diff_distributions),
                                     maxiters=10000,verbose=false)
bound1 = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5)]
 result = bboptimize(obj;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5,1.0] atol = 1e-1

 
data_distributions = [fit_mle(Normal,aggregate_data[i,j,:]) for i in 1:2, j in 1:200]
diff_distributions = [fit_mle(Normal,aggregate_data[i,j,:]-aggregate_data[i,j-1,:]) for j in 2:200, i in 1:2 ]
obj = build_loss_objective(prob1,Tsit5(),LogLikeLoss(t,data_distributions,diff_distributions,0.3),
                                     maxiters=10000,verbose=false)
bound1 = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5)]
result = bboptimize(obj;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5,1.0] atol = 1e-1


distributions = [fit_mle(MvNormal,aggregate_data[:,j,:]) for j in 1:200]
diff_distributions = [fit_mle(MvNormal,aggregate_data[:,j,:]-aggregate_data[:,j-1,:]) for j in 2:200]
priors = [Truncated(Normal(1.5,0.1),0,2),Truncated(Normal(1.0,0.1),0,1.5)]
obj = build_loss_objective(prob1,Tsit5(),LogLikeLoss(t,distributions,diff_distributions),
                                     maxiters=10000,verbose=false,priors=priors)
bound1 = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5)]
result = bboptimize(obj;SearchRange = bound1, MaxSteps = 11e3)
@test result.archive_output.best_candidate ≈ [1.5,1.0] atol = 1e-1
