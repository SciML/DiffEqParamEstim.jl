# DiffEqParamEstim.jl

DiffEqParamEstim.jl is a component package in the DifferentialEquations ecosystem. It holds the
parameter estimation tools. While completely independent
and usable on its own, users interested in using this
functionality should check out [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).

## Installation

This functionality does not come standard with DifferentialEquations.jl.
To use this functionality, you must install DiffEqParamEstim.jl:

```julia
]add DiffEqParamEstim
using DiffEqParamEstim
```

## Recommended Methods

The recommended method is to use `build_loss_objective` with the optimizer
of your choice. This method can thus be paired with global optimizers
from packages like BlackBoxOptim.jl or NLopt.jl which can be much less prone to
finding local minima than local optimization methods. Also, it allows the user
to define the cost function in the way they choose as a function
`loss(sol)`, and thus can fit using any cost function on the solution,
making it applicable to fitting non-temporal data and other types of
problems. Also, `build_loss_objective` works for all of the `DEProblem`
types, allowing it to optimize parameters on ODEs, SDEs, DDEs, DAEs,
etc.

However, this method requires repeated solution of the differential equation.
If the data is temporal data, the most efficient method is the
`two_stage_method` which does not require repeated solutions but is not as
accurate. Usage of the `two_stage_method` should have a post-processing step
which refines using a method like `build_loss_objective`.

## Optimization-Based Methods

### Nonlinear Regression

`build_loss_objective` builds an objective function to be used with Optim.jl
and MathProgBase-associated solvers like NLopt.

```julia
function build_loss_objective(prob::DEProblem,alg,loss_func
                              regularization=nothing;
                              mpg_autodiff = false,
                              verbose_opt = false,
                              verbose_steps = 100,
                              prob_generator = (prob,p)->remake(prob,p=p),
                              kwargs...)
```

The first argument is the `DEProblem` to solve, and next is the `alg` to use.
The `alg` must match the problem type, which can be any `DEProblem`
(ODEs, SDEs, DAEs, DDEs, etc.). `regularization` defaults to nothing
which has no regularization function. One can also choose `verbose_opt` and
`verbose_steps`, which, in the optimization routines, will print the steps
and the values at the steps every `verbose_steps` steps. `mpg_autodiff` uses
autodifferentiation to define the derivative for the MathProgBase solver.
The extra keyword arguments are passed to the differential equation solver.

### Multiple Shooting objective

Multiple Shooting is generally used in Boundary Value Problems (BVP) and is
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
function multiple_shooting_objective(prob::DiffEqBase.DEProblem,alg,loss,
                              regularization=nothing;prior=nothing,
                              mpg_autodiff = false,discontinuity_weight=1.0,
                              verbose_opt = false,
                              prob_generator = STANDARD_PROB_GENERATOR,
                              autodiff_prototype = mpg_autodiff ? zeros(init_N_params) : nothing,
                              autodiff_chunk = mpg_autodiff ? ForwardDiff.Chunk(autodiff_prototype) : nothing,
                              kwargs...)
```

For consistency `multiple_shooting_objective` takes exactly the same arguments
as `build_loss_objective`. It also has the option for `discontinuity_error` as
a keyword argument which assigns weight to the error occurring due to the
discontinuity that arises from the breaking up of the time span.

### Two Stage method (Non-Parametric Collocation)

The two-stage method is a collocation method for estimating parameters without
requiring repeated solving of the differential equation. It does so by
determining a smoothed estimated trajectory of the data (local quadratic polynomial
fit by least squares) and optimizing
the derivative function and the data's timepoints to match the derivatives
of the smoothed trajectory. This method has less accuracy than other methods
but is much faster, and is a good method to try first to get in the general
"good parameter" region, to then finish using one of the other methods.

```julia
function two_stage_method(prob::DEProblem,tpoints,data;kernel= :Epanechnikov,
                          loss_func = L2DistLoss,mpg_autodiff = false,
                          verbose = false,verbose_steps = 100)
```

#### The Loss Function

```julia
loss_func(sol)
```

is a function which reduces the problem's solution to a scalar which the
optimizer will try to minimize. While this is very
flexible, two convenience routines are included for fitting to data with standard
cost functions:

```julia
L2Loss(t,data;differ_weight=nothing,data_weight=nothing,
              colloc_grad=nothing,dudt=nothing)
```

where `t` is the set of timepoints which the data is found at, and
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
LogLikeLoss(t,distributions,diff_distributions=nothing)
```

In this case, there are two forms. The simple case is where `distributions[i,j]`
is the likelihood distributions from a `UnivariateDistribution` from
[Distributions.jl](https://juliastats.github.io/Distributions.jl/dev/), where it
corresponds to the likelihood at `t[i]` for component `j`. The second case is
where `distributions[i]` is a `MultivariateDistribution` which corresponds to
the likelihood at `t[i]` over the vector of components. This likelihood function
then calculates the negative of the total loglikelihood over time as its objective
value (negative since optimizers generally find minimums, and thus this corresponds
to maximum likelihood estimation). The third term, `diff_distributions`, acts
similarly but allows putting a distribution on the first difference terms
`sol[i+1]-sol[i]`.

Note that these distributions can be generated via `fit_mle` on some dataset
against some chosen distribution type.

#### Note About Loss Functions

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
   if any((s.retcode != :Success for s in sol))
     tot_loss = Inf
   else
     # calculation for the loss here
   end
   tot_loss
end
```

#### Note on First Differencing

```julia
L2Loss(t,data,differ_weight=0.3,data_weight=0.7)
```

First differencing incorporates the differences of data points at consecutive
time points which adds more information about the trajectory in the loss
function. Adding first differencing is helpful in cases where the `L2Loss`
alone leads to non-identifiable parameters but adding a first differencing
term makes it more identifiable. This can be noted on stochastic differential
equation models, where this aims to capture the autocorrelation and therefore
helps us avoid getting the same stationary distribution despite different
trajectories and thus wrong parameter estimates.

#### The Regularization Function

The regularization can be any function of `p`, the parameter vector:

```julia
regularization(p)
```

The `Regularization` helper function builds a regularization using a
penalty function `penalty` from
[PenaltyFunctions.jl](https://github.com/JuliaML/PenaltyFunctions.jl):

```julia
Regularization(λ,penalty=L2Penalty())
```

The regularization defaults to L2 if no penalty function is specified.
`λ` is the weight parameter for the addition of the regularization term.

#### The Problem Generator Function

The argument `prob_generator` allows one to specify a function for generating
new problems from a given parameter set. By default, this just builds a new
problem which fixes the element types in a way that's autodifferentiation
compatible and adds the new parameter vector `p`. For example, the code for this is:

```julia
prob_generator = (prob,p) -> remake(prob,u0=convert.(eltype(p),prob.u0),p=p)
```

Then the new problem with these new values is returned.

One can use this to change the meaning of the parameters using this function. For
example, if one instead wanted to optimize the initial conditions for a function
without parameters, you could change this to:

```julia
prob_generator = (prob,p) -> remake(prob.f,u0=p)
```

which simply uses `p` as the initial condition in the initial value problem.

### LeastSquaresOptim.jl objective

`build_lsoptim_objective` builds an objective function to be used with LeastSquaresOptim.jl.

```julia
build_lsoptim_objective(prob,tspan,t,data;
                        prob_generator = (prob,p) -> remake(prob,u0=convert.(eltype(p),prob.u0),p=p),
                        kwargs...)
```

The arguments are the same as `build_loss_objective`.

### lm_fit

`lm_fit` is a function for fitting the parameters of an ODE using the Levenberg-Marquardt
algorithm. This algorithm is really bad and thus not recommended since, for example,
the Optim.jl algorithms on an L2 loss are more performant and robust. However,
this is provided for completeness as most other differential equation libraries
use an LM-based algorithm, so this allows one to test the increased effectiveness
of not using LM.

```julia
lm_fit(prob::DEProblem,tspan,t,data,p0;
       prob_generator = (prob,p) -> remake(prob,u0=convert.(eltype(p),prob.u0),p=p),
       kwargs...)
```

The arguments are similar to before, but with `p0` being the initial conditions
for the parameters and the `kwargs` as the args passed to the LsqFit `curve_fit`
function (which is used for the LM solver). This returns the fitted parameters.

### MAP estimate

You can also add a prior option to `build_loss_objective` and `multiple_shooting_objective` that essentially turns it into MAP by multiplying the loglikelihood (the cost) by the prior. The option was added as a keyword argument `priors`, it can take in either an array of univariate distributions for each of the parameters or a multivariate distribution.

```julia
ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);priors=priors,discontinuity_weight=1.0,abstol=1e-12,reltol=1e-12)
```

## Optimization-Based ODE Inference Examples

### Simple Local Optimization

We choose to optimize the parameters on the Lotka-Volterra equation. We do so
by defining the function as a function with parameters:

```julia
function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
```

We create data using the numerical result with `a=1.5`:

```julia
sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=200))
using RecursiveArrayTools # for VectorOfArray
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
```

Here we used `VectorOfArray` from [RecursiveArrayTools.jl](https://github.com/ChrisRackauckas/RecursiveArrayTools.jl)
to turn the result of an ODE into a matrix.

If we plot the solution with the parameter at `a=1.42`, we get the following:

![Parameter Estimation Not Fit](assets/paramest_notfit.png)

Notice that after one period this solution begins to drift very far off: this
problem is sensitive to the choice of `a`.

To build the objective function for Optim.jl, we simply call the `build_loss_objective`
function:

```julia
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
```

This objective function internally is calling the ODE solver to get solutions
to test against the data. The keyword arguments are passed directly to the solver.
Note that we set `maxiters` in a way that causes the differential equation solvers to
error more quickly when in bad regions of the parameter space, speeding up the
process. If the integrator stops early (due to divergence), then those parameters
are given an infinite loss, and thus this is a quick way to avoid bad parameters.
We set `verbose=false` because this divergence can get noisy.

Before optimizing, let's visualize our cost function by plotting it for a range
of parameter values:

```julia
vals = 0.0:0.1:10.0
using Plots; plotly()
plot(vals,[cost_function(i) for i in vals],yscale=:log10,
     xaxis = "Parameter", yaxis = "Cost", title = "1-Parameter Cost Function",
     lw = 3)
```

![1 Parameter Likelihood](assets/1paramcost.png)

Here we see that there is a very well-defined minimum in our cost function at
the real parameter (because this is where the solution almost exactly fits the
dataset).

Now this cost function can be used with Optim.jl in order to get the parameters.
For example, we can use Brent's algorithm to search for the best solution on
the interval `[0,10]` by:

```julia
using Optim
result = optimize(cost_function, 0.0, 10.0)
```

This returns `result.minimizer[1]==1.5` as the best parameter to match the data.
When we plot the fitted equation on the data, we receive the following:

![Parameter Estimation Fit](assets/paramest_fit.png)

Thus we see that after fitting, the lines match up with the generated data and
receive the right parameter value.

We can also use the multivariate optimization functions. For example, we can use
the `BFGS` algorithm to optimize the parameter starting at `a=1.42` using:

```julia
result = optimize(cost_function, [1.42], BFGS())
```

Note that some of the algorithms may be sensitive to the initial condition. For more
details on using Optim.jl, see the [documentation for Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/).
We can improve our solution by noting that the Lotka-Volterra equation requires that
the parameters are positive. Thus [following the Optim.jl documentation](https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#box-constrained-optimization)
we can add box constraints to ensure the optimizer only checks between 0.0 and 3.0
which improves the efficiency of our algorithm:

```julia
lower = [0.0]
upper = [3.0]
result = optimize(cost_function, lower, upper, [1.42], Fminbox(BFGS()))
```

Lastly, we can use the same tools to estimate multiple parameters simultaneously.
Let's use the Lotka-Volterra equation with all parameters free:

```julia
function f2(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f2,u0,tspan,p)
```

We can build an objective function and solve the multiple parameter version just as before:

```julia
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
```
We can also use First-Differences in L2Loss by passing the kwarg `differ_weight` which decides the contribution of the
differencing loss to the total loss.

```julia
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data,differ_weight=0.3,data_weight=0.7),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function, [1.3,0.8,2.8,1.2], Optim.BFGS())
```

To solve it using LeastSquaresOptim.jl, we use the `build_lsoptim_objective` function:

```julia
cost_function = build_lsoptim_objective(prob1,t,data,Tsit5())
```

The result is a cost function which can be used with LeastSquaresOptim. For more
details, consult the [documentation for LeastSquaresOptim.jl](https://github.com/matthieugomez/LeastSquaresOptim.jl):

```julia
x = [1.3,0.8,2.8,1.2]
res = optimize!(LeastSquaresProblem(x = x, f! = cost_function,
                output_length = length(t)*length(prob.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR())
```

We can see the results are:

```julia
println(res.minimizer)

Results of Optimization Algorithm
 * Algorithm: Dogleg
 * Minimizer: [1.4995074428834114,0.9996531871795851,3.001556360700904,1.0006272074128821]
 * Sum of squares at Minimum: 0.035730
 * Iterations: 63
 * Convergence: true
 * |x - x'| < 1.0e-15: true
 * |f(x) - f(x')| / |f(x)| < 1.0e-14: false
 * |g(x)| < 1.0e-14: false
 * Function Calls: 64
 * Gradient Calls: 9
 * Multiplication Calls: 135
```

and thus this algorithm was able to correctly identify all four parameters.

We can also use Multiple Shooting method by creating a `multiple_shooting_objective`

```julia
function ms_f(du,u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -3*u[2] + u[1]*u[2]
end
ms_u0 = [1.0;1.0]
tspan = (0.0,10.0)
ms_p = [1.5,1.0]
ms_prob = ODEProblem(ms_f,ms_u0,tspan,ms_p)
t = collect(range(0,stop=10,length=200))
data = Array(solve(ms_prob,Tsit5(),saveat=t,abstol=1e-12,reltol=1e-12))
bound = Tuple{Float64, Float64}[(0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),
                                (0, 10),(0, 10),(0, 10),(0, 10),(0, 10),(0, 10)]


ms_obj = multiple_shooting_objective(ms_prob,Tsit5(),L2Loss(t,data);discontinuity_weight=1.0,abstol=1e-12,reltol=1e-12)
```

This creates the objective function that can be passed to an optimizer from which we can then get the parameter values
and the initial values of the short time periods keeping in mind the indexing.

```julia
result = bboptimize(ms_obj;SearchRange = bound, MaxSteps = 21e3)
result.archive_output.best_candidate[end-1:end]
```
Giving us the results as
```julia
Starting optimization with optimizer BlackBoxOptim.DiffEvoOpt{BlackBoxOptim.FitPopulation{Float64},BlackBoxOptim.RadiusLimitedSelector,BlackBoxOptim.AdaptiveDiffEvoRandBin{3},BlackBoxOptim.RandomBound{BlackBoxOptim.RangePerDimSearchSpace}}

Optimization stopped after 21001 steps and 136.60030698776245 seconds
Termination reason: Max number of steps (21000) reached
Steps per second = 153.7405036862868
Function evals per second = 154.43596332393247
Improvements/step = 0.17552380952380953
Total function evaluations = 21096


Best candidate found: [0.997396, 1.04664, 3.77834, 0.275823, 2.14966, 4.33106, 1.43777, 0.468442, 6.22221, 0.673358, 0.970036, 2.05182, 2.4216, 0.274394, 5.64131, 3.38132, 1.52826, 1.01721]

Fitness: 0.126884213

Out[4]:2-element Array{Float64,1}:
        1.52826
        1.01721
```
Here as our model had 2 parameters, we look at the last two indexes of `result` to get our parameter values and
the rest of the values are the initial values of the shorter timespans as described in the reference section.

The objective function for Two Stage method can be created and passed to an optimizer as

```julia
two_stage_obj = two_stage_method(ms_prob,t,data)
result = Optim.optimize(two_stage_obj, [1.3,0.8,2.8,1.2], Optim.BFGS()
)
Results of Optimization Algorithm
 * Algorithm: BFGS
 * Starting Point: [1.3,0.8,2.8,1.2]
 * Minimizer: [1.5035938533664717,0.9925731153746833, ...]
 * Minimum: 1.513400e+00
 * Iterations: 9
 * Convergence: true
   * |x - x'| ≤ 0.0e+00: false
     |x - x'| = 4.58e-10
   * |f(x) - f(x')| ≤ 0.0e+00 |f(x)|: false
     |f(x) - f(x')| = 5.87e-16 |f(x)|
   * |g(x)| ≤ 1.0e-08: true
     |g(x)| = 7.32e-11
   * Stopped by an increasing objective: false
   * Reached Maximum Number of Iterations: false
 * Objective Calls: 31
 * Gradient Calls: 31
```
The default kernel used in the method is `Epanechnikov` others that are available are `Uniform`,  `Triangular`,
`Quartic`, `Triweight`, `Tricube`, `Gaussian`, `Cosine`, `Logistic` and `Sigmoid`, this can be passed by the
`kernel` keyword argument. `loss_func` keyword argument can be used to pass the loss function (cost function) you want
 to use and `mpg_autodiff` enables Auto Differentiation.

### More Algorithms (Global Optimization) via MathProgBase Solvers

The `build_loss_objective` function builds an objective function which is able
to be used with MathProgBase-associated solvers. This includes packages like
IPOPT, NLopt, MOSEK, etc. Building off of the previous example, we can build a
cost function for the single parameter optimization problem like:

```julia
function f(du,u,p,t)
  dx = p[1]*u[1] - u[1]*u[2]
  dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5())

t = collect(range(0,stop=10,length=200))
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)

obj = build_loss_objective(prob,Tsit5(),L2Loss(t,data),maxiters=10000)
```

We can now use this `obj` as the objective function with MathProgBase solvers.
For our example, we will use NLopt. To use the local derivative-free
Constrained Optimization BY Linear Approximations algorithm, we can simply do:

```julia
using NLopt
opt = Opt(:LN_COBYLA, 1)
min_objective!(opt, obj)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
```

This finds a minimum at `[1.49997]`. For a modified evolutionary algorithm, we
can use:

```julia
opt = Opt(:GN_ESCH, 1)
min_objective!(opt, obj)
lower_bounds!(opt,[0.0])
upper_bounds!(opt,[5.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 100000)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
```

We can even use things like the Improved Stochastic Ranking Evolution Strategy
(and add constraints if needed). This is done via:

```julia
opt = Opt(:GN_ISRES, 1)
min_objective!(opt, obj.cost_function2)
lower_bounds!(opt,[-1.0])
upper_bounds!(opt,[5.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 100000)
(minf,minx,ret) = NLopt.optimize(opt,[0.2])
```

which is very robust to the initial condition. The fastest result comes from the
following:

```julia
using NLopt
opt = Opt(:LN_BOBYQA, 1)
min_objective!(opt, obj)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
```

For more information, see the NLopt documentation for more details. And give IPOPT
or MOSEK a try!

### Using JuMP with DiffEqParamEstim

[JuMP](https://github.com/JuliaOpt/JuMP.jl) is a domain-specific modeling language
for mathematical optimization embedded in Julia.

```julia
using OrdinaryDiffEq, DiffEqParamEstim, JuMP, NLopt, Plots
```
Let's define the Lorenz equation to use as our example

```julia
function g(du,u,p,t)
  σ,ρ,β = p
  x,y,z = u
  du[1] = dx = σ*(y-x)
  du[2] = dy = x*(ρ-z) - y
  du[3] = dz = x*y - β*z
end
```

Let's get a solution of the system with parameter values `σ=10.0` `ρ=28.0` `β=8/3` to use as our
data. We define some convenience functions `model_ode` (to create an `ODEProblem`) and `solve_model`(to obtain
solution of the `ODEProblem`) to use in a custom objective function later.

```julia
u0 = [1.0;0.0;0.0]
t = 0.0:0.01:1.0
tspan = (0.0,1.0)
model_ode(p_) = ODEProblem(g, u0, tspan,p_)
solve_model(mp_) = OrdinaryDiffEq.solve(model_ode(mp_), Tsit5(),saveat=0.01)
mock_data = Array(solve_model([10.0,28.0,8/3]))
```
Now we define a custom objective function to pass for optimization to JuMP using
the `build_loss_objective` described above provided by DiffEqParamEstim that defines an objective
function for the parameter estimation problem.

```julia
loss_objective(mp_, dat) = build_loss_objective(model_ode(mp_), Tsit5(), L2Loss(t,dat))
```

We create a JuMP model, variables, set the objective function and the choice of
optimization algorithm to be used in the JuMP syntax. You can read more about this in
JuMP's [documentation](http://www.juliaopt.org/JuMP.jl/0.18/index.html).

```julia
juobj(args...) = loss_objective(args, mock_data)(args)
jumodel = Model()
JuMP.register(jumodel, :juobj, 3, juobj, autodiff=true)
@variables jumodel begin
    σ,(start=8)
    ρ,(start=25.0)
    β,(start=10/3)
end
@NLobjective(jumodel, Min, juobj(σ, ρ, β))
setsolver(jumodel, NLoptSolver(algorithm=:LD_MMA))
```

Let's call the optimizer to obtain the fitted parameter values.

```julia
sol = JuMP.solve(jumodel)
best_mp = getvalue.(getindex.((jumodel,), Symbol.(jumodel.colNames)))
```
Let's compare the solution at the obtained parameter values and our data.

```julia
sol = OrdinaryDiffEq.solve(best_mp |> model_ode, Tsit5())
plot(getindex.(sol.(t),1))
scatter!(mock_data, markersize=2)
```
![jumpestimationplot](assets/jumpestimationplot.png)
### Generalized Likelihood Example

In this example we will demo the likelihood-based approach to parameter fitting.
First let's generate a dataset to fit. We will re-use the Lotka-Volterra equation
but in this case fit just two parameters.

```julia
f1 = function (du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3.0 * u[2] + u[1]*u[2]
end
p = [1.5,1.0]
u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob1 = ODEProblem(f1,u0,tspan,p)
sol = solve(prob1,Tsit5())
```

This is a function with two parameters, `[1.5,1.0]` which generates the same
ODE solution as before. This time, let's generate 100 datasets where at each point
adds a little bit of randomness:

```julia
using RecursiveArrayTools # for VectorOfArray
t = collect(range(0,stop=10,length=200))
function generate_data(sol,t)
  randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
  data = convert(Array,randomized)
end
aggregate_data = convert(Array,VectorOfArray([generate_data(sol,t) for i in 1:100]))
```

here with `t` we measure the solution at 200 evenly spaced points. Thus `aggregate_data`
is a 2x200x100 matrix where `aggregate_data[i,j,k]` is the `i`th component at time
`j` of the `k`th dataset. What we first want to do is get a matrix of distributions
where `distributions[i,j]` is the likelihood of component `i` at take `j`. We
can do this via `fit_mle` on a chosen distributional form. For simplicity we
choose the `Normal` distribution. `aggregate_data[i,j,:]` is the array of points
at the given component and time, and thus we find the distribution parameters
which fits best at each time point via:

```julia
using Distributions
distributions = [fit_mle(Normal,aggregate_data[i,j,:]) for i in 1:2, j in 1:200]
```

Notice for example that we have:

```julia
julia> distributions[1,1]
Distributions.Normal{Float64}(μ=1.0022440583676806, σ=0.009851964521952437)
```

that is, it fit the distribution to have its mean just about where our original
solution was and the variance is about how much noise we added to the dataset.
This this is a good check to see that the distributions we are trying to fit
our parameters to makes sense.

Note that in this case the `Normal` distribution was a good choice, and in many
cases it's a nice go-to choice, but one should experiment with other choices
of distributions as well. For example, a `TDist` can be an interesting way to
incorporate robustness to outliers since low degrees of free T-distributions
act like Normal distributions but with longer tails (though `fit_mle` does not
work with a T-distribution, you can get the means/variances and build appropriate
distribution objects yourself).

Once we have the matrix of distributions, we can build the objective function
corresponding to that distribution fit:

```julia
using DiffEqParamEstim
obj = build_loss_objective(prob1,Tsit5(),LogLikeLoss(t,distributions),
                                     maxiters=10000,verbose=false)
```

First let's use the objective function to plot the likelihood landscape:

```julia
using Plots; plotly()
prange = 0.5:0.1:5.0
heatmap(prange,prange,[obj([j,i]) for i in prange, j in prange],
        yscale=:log10,xlabel="Parameter 1",ylabel="Parameter 2",
        title="Likelihood Landscape")
```

![2 Parameter Likelihood](assets/2paramlike.png)

Recall that this is the negative loglikelihood and thus the minimum is the
maximum of the likelihood. There is a clear valley where the first parameter
is 1.5, while the second parameter's likelihood is more muddled. By taking a
one-dimensional slice:

```julia
plot(prange,[obj([1.5,i]) for i in prange],lw=3,
     title="Parameter 2 Likelihood (Parameter 1 = 1.5)",
     xlabel = "Parameter 2", ylabel = "Objective Function Value")
```

![1 Parameter Likelihood](assets/1paramlike.png)

we can see that there's still a clear minimum at the true value. Thus we will
use the global optimizers from BlackBoxOptim.jl to find the values. We set our
search range to be from `0.5` to `5.0` for both of the parameters and let it
optimize:

```julia
using BlackBoxOptim
bound1 = Tuple{Float64, Float64}[(0.5, 5),(0.5, 5)]
result = bboptimize(obj;SearchRange = bound1, MaxSteps = 11e3)

Starting optimization with optimizer BlackBoxOptim.DiffEvoOpt{BlackBoxOptim.FitPopulation{Float64},B
lackBoxOptim.RadiusLimitedSelector,BlackBoxOptim.AdaptiveDiffEvoRandBin{3},BlackBoxOptim.RandomBound
{BlackBoxOptim.RangePerDimSearchSpace}}
0.00 secs, 0 evals, 0 steps
0.50 secs, 1972 evals, 1865 steps, improv/step: 0.266 (last = 0.2665), fitness=-737.311433781
1.00 secs, 3859 evals, 3753 steps, improv/step: 0.279 (last = 0.2913), fitness=-739.658421879
1.50 secs, 5904 evals, 5799 steps, improv/step: 0.280 (last = 0.2830), fitness=-739.658433715
2.00 secs, 7916 evals, 7811 steps, improv/step: 0.225 (last = 0.0646), fitness=-739.658433715
2.50 secs, 9966 evals, 9861 steps, improv/step: 0.183 (last = 0.0220), fitness=-739.658433715

Optimization stopped after 11001 steps and 2.7839999198913574 seconds
Termination reason: Max number of steps (11000) reached
Steps per second = 3951.50873439296
Function evals per second = 3989.2242527195904
Improvements/step = 0.165
Total function evaluations = 11106


Best candidate found: [1.50001, 1.00001]

Fitness: -739.658433715
```

This shows that it found the true parameters as the best fit to the likelihood.

## Parameter Estimation for Stochastic Differential Equations and Ensembles

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
