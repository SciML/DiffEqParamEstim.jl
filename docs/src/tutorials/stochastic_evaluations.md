# Parameter Estimation for Stochastic Differential Equations and Ensembles

We can use any `DEProblem`, which not only includes `DAEProblem` and `DDEProblem`s,
but also stochastic problems. In this case, let's use the generalized maximum
likelihood to fit the parameters of an SDE's ensemble evaluation.

Let's use the same Lotka-Volterra equation as before, but this time add noise:

```@example sde
using DifferentialEquations, DiffEqParamEstim, Plots, Optimization, ForwardDiff, OptimizationOptimJL

pf_func = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0]
pg_func = function (du,u,p,t)
  du[1] = 1e-6u[1]
  du[2] = 1e-6u[2]
end
prob = SDEProblem(pf_func,pg_func,u0,tspan,p)
sol = solve(prob,SRIW1())
```

Now let's generate a dataset from 10,000 solutions of the SDE

```@example sde
using RecursiveArrayTools # for VectorOfArray
t = collect(range(0, stop=10, length=200))
function generate_data(t)
  sol = solve(prob,SRIW1())
  randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
  data = convert(Array,randomized)
end
aggregate_data = convert(Array,VectorOfArray([generate_data(t) for i in 1:10000]))
```

Now let's estimate the parameters. Instead of using single runs from the SDE, we
will use a `EnsembleProblem`. This means that it will solve the SDE `N` times
to come up with an approximate probability distribution at each time point and
use that in the likelihood estimate.

```@example sde
monte_prob = EnsembleProblem(prob)
```

We use Optim.jl for optimization below

```@example sde
obj = build_loss_objective(monte_prob,SOSRI(),L2Loss(t,aggregate_data), Optimization.AutoForwardDiff(),
                                     maxiters=10000,verbose=false,trajectories = 1000)
optprob = Optimization.OptimizationProblem(obj, [1.0, 0.5])
result = solve(optprob, Optim.BFGS())
```

Parameter Estimation in case of SDE's with a regular `L2Loss` can have poor accuracy due to only fitting against the mean properties as mentioned in [First Differencing](http://docs.juliadiffeq.org/dev/analysis/parameter_estimation/#First-differencing-1).

```@example sde
result.original
```

Instead, when we use `L2Loss` with first differencing enabled, we get much more accurate estimates.

```@example sde
obj = build_loss_objective(monte_prob,SRIW1(),L2Loss(t,aggregate_data,differ_weight=1.0,data_weight=0.5), Optimization.AutoForwardDiff(),
                                  verbose=false, trajectories = 1000, maxiters=1000)
optprob = Optimization.OptimizationProblem(obj, [1.0, 0.5])
result = solve(optprob, Optim.BFGS())
result.original
```

Here, we see that we successfully recovered the drift parameter, and got close to
the original noise parameter after searching a two-orders-of-magnitude range.
