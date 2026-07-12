# Parameter Estimation for Stochastic Differential Equations and Ensembles

We can use any `DEProblem`, which not only includes `DAEProblem` and `DDEProblem`s,
but also stochastic problems. In this case, let's use the generalized maximum
likelihood to fit the parameters of an SDE's ensemble evaluation.

Let's use the same Lotka-Volterra equation as before, but this time add noise:

```@example sde
using DifferentialEquations, DiffEqParamEstim, Plots, Optimization, ForwardDiff,
      OptimizationOptimJL
using Random, Statistics
using StochasticDiffEq: SRIW1
using SciMLLogging: None

pf_func = function (du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0]
pg_func = function (du, u, p, t)
    du[1] = 1e-6u[1]
    du[2] = 1e-6u[2]
end
prob = SDEProblem(pf_func, pg_func, u0, tspan, p)
sol = solve(prob, SRIW1())
```

Now let's generate a compact dataset from repeated solutions of the SDE:

```@example sde
using RecursiveArrayTools # for VectorOfArray
t = collect(range(0, stop = 10, length = 200))
Random.seed!(1234)
function generate_data(t)
    sol = solve(prob, SRIW1())
    randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
    data = convert(Array, randomized)
end
aggregate_data = convert(Array, VectorOfArray([generate_data(t) for i in 1:10]))
data = dropdims(mean(aggregate_data; dims = 3); dims = 3)
```

Now let's estimate the parameters. Instead of using single runs from the SDE, we
will use a `EnsembleProblem`. This means that it will solve the SDE `N` times
to come up with an approximate probability distribution at each time point and
use that in the likelihood estimate.

```@example sde
monte_prob = EnsembleProblem(prob)
function seeded_prob_generator(_, parameters)
    Random.seed!(2468)
    candidate_prob = remake(prob; u0 = eltype(parameters).(u0), p = parameters)
    return EnsembleProblem(candidate_prob)
end
```

The objective uses common random numbers so that every candidate is evaluated against
the same simulated noise paths. This makes the objective deterministic enough for a
gradient-based optimizer while retaining an ensemble evaluation.

We use Optim.jl for optimization below

```@example sde
obj = build_loss_objective(monte_prob, SRIW1(), L2Loss(t, data),
    Optimization.AutoForwardDiff(),
    maxiters = 1000, verbose = None(), trajectories = 10,
    prob_generator = seeded_prob_generator)
lower = zeros(2)
upper = fill(2.0, 2)
optprob = Optimization.OptimizationProblem(obj, [1.4, 0.95], lb = lower, ub = upper)
result = solve(optprob, Optim.BFGS())
@assert isapprox(result.u, p; atol = 0.05)
```

This regular `L2Loss` fits the mean state values. First differencing offers an
alternative that also fits the increments between observations, as described in
[First Differencing](@ref fd).

```@example sde
result.original
```

We enable first differencing by assigning weights to the state and increment residuals.

```@example sde
obj = build_loss_objective(monte_prob, SRIW1(),
    L2Loss(t, data, differ_weight = 1.0,
        data_weight = 0.5), Optimization.AutoForwardDiff(),
    verbose = None(), trajectories = 10, maxiters = 1000,
    prob_generator = seeded_prob_generator)
optprob = Optimization.OptimizationProblem(obj, [1.4, 0.95], lb = lower, ub = upper)
result = solve(optprob, Optim.BFGS())
@assert isapprox(result.u, p; atol = 0.05)
result.original
```

Both drift parameters are recovered from the synthetic data. Larger ensembles
reduce the Monte Carlo error in the objective.

```@example sde
println(result.u)
```
