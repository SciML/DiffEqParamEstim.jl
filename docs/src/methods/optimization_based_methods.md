# Optimization-Based Methods

## The Objective Function Builders

### Standard Nonlinear Regression

`build_loss_objective` builds an objective function to be used with Optim.jl
and MathProgBase-associated solvers like NLopt.

```julia
function build_loss_objective(prob::DEProblem, alg, loss,
                              adtype = SciMLBase.NoAD(),
                              regularization = nothing;
                              priors = nothing,
                              prob_generator = STANDARD_PROB_GENERATOR,
                              kwargs...)
end
```

The first argument is the `DEProblem` to solve, and next is the `alg` to use.
The `alg` must match the problem type, which can be any `DEProblem`
(ODEs, SDEs, DAEs, DDEs, etc.). `regularization` defaults to nothing, which has no regularization function.
The extra keyword arguments are passed to the differential equation solver.

### Multiple Shooting

Multiple Shooting is often used in Boundary Value Problems (BVP) and is
more robust than the regular objective function used in these problems. It
proceeds as follows:

  - Divide up the time span into short time periods and solve the equation
    with the current parameters which here consist of both, the parameters of the
    differential equations and also the initial values for the short time periods.
  - This objective additionally involves a discontinuity error term that imposes
    higher cost if the end of the solution of one time period doesn't match the
    beginning of the next one.
  - Merge the solutions from the shorter intervals and then calculate the loss.

```julia
function multiple_shooting_objective(prob::DiffEqBase.DEProblem, alg, loss,
                                     adtype = SciMLBase.NoAD(),
                                     regularization = nothing;
                                     priors = nothing,
                                     discontinuity_weight = 1.0,
                                     prob_generator = STANDARD_PROB_GENERATOR,
                                     kwargs...)
end
```

For consistency `multiple_shooting_objective` takes exactly the same arguments
as `build_loss_objective`. It also has the option for `discontinuity_weight` as
a keyword argument, which assigns weight to the error occurring due to the
discontinuity that arises from the breaking up of the time span.

## Detailed Explanations of Arguments

### The Loss Function

```julia
loss(sol)
```

is the function, which reduces the problem's solution to a scalar, which the
optimizer will try to minimize. While this is very
flexible, two convenience routines are included for fitting to data with standard
cost functions:

```julia
L2Loss(t, data; differ_weight = nothing, data_weight = nothing,
       colloc_grad = nothing, dudt = nothing)
```

where `t` is the set of timepoints which the data are found at, and
`data` are the values that are known where each column corresponds to measures
of the values of the system. `L2Loss` is an optimized version
of the L2-distance. The `data_weight` is a scalar or vector
of weights for the loss function which must match the size of the data.
Note that minimization of a weighted `L2Loss` is equivalent to maximum
likelihood estimation of a heteroskedastic Normally distributed likelihood.
`differ_weight` allows one to add a weight on the first differencing terms
`sol[i+1]-sol[i]` against the data first differences. This smooths out the
loss term and can make it easier to fit strong solutions of stochastic models,
but is zero (nothing) by default. Additionally, `colloc_grad` allows one to
give a matrix of the collocation gradients for the data. This is used to add
an interpolation derivative term, like the two-stage method. A convenience
function `colloc_grad(t,data)` returns a collocation gradient from a 3rd order
spline calculated by Dierckx.jl, which can be used as the `colloc_grad`. Note
that, with a collocation gradient and regularization, this loss is equivalent
to a 4DVAR.

Additionally, we include a more flexible log-likelihood approach:

```julia
LogLikeLoss(t, distributions, diff_distributions = nothing)
```

In this case, there are two forms. The simple case is where `distributions[i,j]`
is the likelihood distributions from a `UnivariateDistribution` from
[Distributions.jl](https://juliastats.github.io/Distributions.jl/dev/), where it
corresponds to the likelihood at `t[i]` for component `j`. The second case is
where `distributions[i]` is a `MultivariateDistribution` which corresponds to
the likelihood at `t[i]` over the vector of components. This likelihood function
then calculates the negative of the total log-likelihood over time as its objective
value (negative since optimizers generally find minimums, and thus this corresponds
to maximum likelihood estimation). The third term, `diff_distributions`, acts
similarly but allows putting a distribution on the first difference terms
`sol[i+1]-sol[i]`.

Note that these distributions can be generated via `fit_mle` on some dataset
against some chosen distribution type.

### Note About Loss Functions

For parameter estimation problems, it's not uncommon for the optimizers to hit
unstable regions of parameter space. This causes warnings that the solver exited
early, and the built-in loss functions like `L2Loss`
automatically handle this. However, if using a user-supplied loss function,
you should make sure it's robust to these issues. One common pattern is to
apply infinite loss when the integration is not successful. Using the retcodes,
this can be done via:

```julia
function my_loss_function(sol)
    tot_loss = 0.0
    if any((!SciMLBase.successful_retcode(s.retcode) for s in sol))
        tot_loss = Inf
    else
        # calculation for the loss here
    end
    tot_loss
end
```

### Note on First Differencing

```julia
L2Loss(t, data, differ_weight = 0.3, data_weight = 0.7)
```

First differencing incorporates the differences of data points at consecutive
time points which adds more information about the trajectory in the loss
function. Adding first differencing is helpful in cases where the `L2Loss`
alone leads to non-identifiable parameters, but adding a first differencing
term makes it more identifiable. This can be noted on stochastic differential
equation models, where this aims to capture the autocorrelation and therefore
helps us avoid getting the same stationary distribution despite different
trajectories and thus wrong parameter estimates.

### The Regularization Function

The regularization can be any function of `p`, the parameter vector:

```julia
regularization(p)
```

The `Regularization` helper function builds a regularization using a
penalty function `penalty` from
[PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl):

```julia
reg = Regularization(λ, penalty = L2Penalty())
build_loss_objective(prob, alg, loss, SciMLBase.NoAD(), reg)

using Optimization, Zygote
build_loss_objective(prob, alg, loss, Optimization.AutoZygote(), reg)
```

The regularization defaults to L2 if no penalty function is specified.
`λ` is the weight parameter for the addition of the regularization term.

### Using automatic differentiation

To use derivatives with optimization solvers, Optimization.jl's
`adtype` argument as described [here](https://docs.sciml.ai/Optimization/stable/tutorials/intro/#Controlling-Gradient-Calculations-(Automatic-Differentiation))
should be used with the wrapper subpackage OptimizationOptimJL, OptimizationNLopt etc.

```julia
using Optimization, ForwardDiff
build_loss_objective(prob, alg, loss, Optimization.AutoForwardDiff())
multiple_shooting_objective(prob, alg, loss, Optimization.AutoForwardDiff())
```

### The Problem Generator Function

The argument `prob_generator` allows one to specify a function for generating
new problems from a given parameter set. By default, this just builds a new
problem which fixes the element types in a way that's autodifferentiation
compatible and adds the new parameter vector `p`. For example, the code for this is:

```julia
prob_generator = (prob, p) -> remake(prob, u0 = convert.(eltype(p), prob.u0), p = p)
```

Then the new problem with these new values is returned.

One can use this to change the meaning of the parameters using this function. For
example, if one instead wanted to optimize the initial conditions for a function
without parameters, you could change this to:

```julia
prob_generator = (prob, p) -> remake(prob.f, u0 = p)
```

which simply uses `p` as the initial condition in the initial value problem.

## Using the Objectives for MAP estimates

You can also add a prior option to `build_loss_objective` and `multiple_shooting_objective` that
essentially turns it into MAP by multiplying the log-likelihood (the cost) by the prior. The option is available
as the keyword argument `priors`, it can take in either an array of univariate distributions for each of
the parameters or a multivariate distribution.

```julia
ms_obj = multiple_shooting_objective(ms_prob, Tsit5(), L2Loss(t, data); priors = priors,
                                     discontinuity_weight = 1.0, abstol = 1e-12,
                                     reltol = 1e-12)
```
