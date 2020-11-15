# Parameter Estimation for Stochastic Differential Equations and Ensembles

We can use any `DEProblem`, which not only includes `DAEProblem` and `DDEProblem`s,
but also stochastic problems. In this case, let's use the generalized maximum
likelihood to fit the parameters of an SDE's ensemble evaluation.

Let's use the same Lotka-Volterra equation as before, but this time add noise:

```julia
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

Now lets generate a dataset from 10,000 solutions of the SDE

```julia
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

```julia
monte_prob = EnsembleProblem(prob)
```

We use Optim.jl for optimization below

```julia
obj = build_loss_objective(monte_prob,SOSRI(),L2Loss(t,aggregate_data),
                                     maxiters=10000,verbose=false,num_monte = 1000,
                                     parallel_type = :threads)
result = Optim.optimize(obj, [1.0,0.5], Optim.BFGS())
```
Parameter Estimation in case of SDE's with a regular `L2Loss` can have poor accuracy due to only fitting against the mean properties as mentioned in [First Differencing](http://docs.juliadiffeq.org/dev/analysis/parameter_estimation/#First-differencing-1).

```julia
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [1.0,0.5]
 * Minimizer: [6.070728870478734,5.113357737345448]
 * Minimum: 1.700440e+03
 * Iterations: 14
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 1.00e-03
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 1.81e-07 |f(x)|
   * |g(x)| ≤ 1.0e-08: false
     |g(x)| = 2.34e+00
   * Stopped by an increasing objective: true
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 61
 * Gradient Calls: 61
```

Instead when we use `L2Loss` with first differencing enabled we get much more accurate estimates.

```julia
 obj = build_loss_objective(monte_prob,SRIW1(),L2Loss(t,data,differ_weight=1.0,data_weight=0.5),maxiters=1000,
                                  verbose=false,verbose_opt=false,verbose_steps=1,num_monte=50)
result = Optim.optimize(obj, [1.0,0.5], Optim.BFGS())
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [1.0,0.5]
 * Minimizer: [1.5010687426045128,1.0023453619050238]
 * Minimum: 1.166650e-01
 * Iterations: 16
 * Convergence: false
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 6.84e-09
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 5.85e-06 |f(x)|
   * |g(x)| ≤ 1.0e-08: false
     |g(x)| = 1.81e-01
   * Stopped by an increasing objective: true
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 118
 * Gradient Calls: 118
```

Here, we see that we successfully recovered the drift parameter, and got close to
the original noise parameter after searching a two-orders-of-magnitude range.
