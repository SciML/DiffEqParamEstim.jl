using DiffEqParamEstim, OrdinaryDiffEq, StochasticDiffEq, ParameterizedFunctions,
      DiffEqBase, RecursiveArrayTools
using Base.Test

pf_func = function (t,u,p,du)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end

pf = ParameterizedFunction(pf_func,[1.5,1.0])
u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(pf,u0,tspan)
sol = solve(prob,Tsit5())

t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]
data = vecvec_to_mat(randomized)

monte_prob = MonteCarloProblem(prob)
obj = build_loss_objective(monte_prob,Tsit5(),CostVData(t,data),maxiters=10000,verbose=false,num_monte=5)

import Optim
result = Optim.optimize(obj, [1.0,0.8], Optim.BFGS())
@test_approx_eq_eps result.minimizer [1.5,1.0] 3e-1

pg_func = function (t,u,p,du)
  du[1] = 1e-6u[1]
  du[2] = 1e-6u[2]
end
pg = ParameterizedFunction(pg_func,[])
tspan = (0.0,10.0)
prob = SDEProblem(pf,pg,u0,tspan)
sol = solve(prob,SRIW1())

monte_prob = MonteCarloProblem(prob)

srand(200)
obj = build_loss_objective(monte_prob,SRIW1(),CostVData(t,data),maxiters=1000,
                           verbose=false,verbose_opt=false,verbose_steps=1,num_monte=25)

result = Optim.optimize(obj, [1.0,0.95], Optim.BFGS())
@test_approx_eq_eps result.minimizer [1.5,1.0] 3e-1
