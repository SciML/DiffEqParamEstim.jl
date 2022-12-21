# Getting Started with Optimization-Based ODE Parameter Estimation

In this tutorial we will showcase how to estimate the parameters of an ordinary
differential equation using DiffEqParamEstim.jl. DiffEqParamEstim.jl is a high
level tool that makes common parameter estimation tasks simple. Here we will show
how to use its cost function generation to estimate the parameters of the Lotka-Volterra
equation against simulated data.

## Installation

First we will make sure DiffEqParamEstim.jl is installed correctly. To do this, we use the
Julia package REPL, opened by typing `]` in the REPL and seeing `pkg>` appear in blue.
Then we type: `add DiffEqParamEstim` and hit enter. This will run the package management
sequence and we will be good to go.

## Required Dependencies

| Module      | Description |
| ----------- | ----------- |
| [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/) | The numerical differential equation solver package |
| [RecursiveArrayTools.jl](https://docs.sciml.ai/RecursiveArrayTools/stable/) | Tooling for recursive arrays like vector of arrays |
| [Plots.jl](https://docs.juliaplots.org/latest/) | Tooling for plotting and visualization |
| [Zygote.jl](https://docs.sciml.ai/Zygote/stable/) | Tooling for reverse-mode automatic differentiation (gradient calculations) |
| [Optimization.jl](https://docs.sciml.ai/Optimization/stable/) | The numerical optimization package |
| [OptimizationOptimJL.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/optim/) | The Optim optimizers we will use for local optimization |
| [OptimizationBBO.jl](https://docs.sciml.ai/Optimization/stable/optimization_packages/blackboxoptim/) | The BlackBoxOptim optimizers we will use for global optimization |

## Parameter Estimation in the Lotka-Volterra Equation: 1 Parameter Case

We choose to optimize the parameters on the Lotka-Volterra equation. Let's start
by defining the equation as a function with parameters:

```@example ode
using DifferentialEquations, RecursiveArrayTools, Plots, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO

function f(du,u,p,t)
  du[1] = dx = p[1]*u[1] - u[1]*u[2]
  du[2] = dy = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
```

### Generating Synthetic Data

We create data using the numerical result with `a=1.5`:

```@example ode
sol = solve(prob,Tsit5())
t = collect(range(0,stop=10,length=200))
using RecursiveArrayTools # for VectorOfArray
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
```

Here we used `VectorOfArray` from [RecursiveArrayTools.jl](https://docs.sciml.ai/RecursiveArrayTools/stable/)
to turn the result of an ODE into a matrix.

If we plot the solution with the parameter at `a=1.42`, we get the following:

```@example ode
newprob = remake(prob, p = [1.42])
newsol = solve(newprob,Tsit5())
plot(sol)
plot!(newsol)
```

Notice that after one period this solution begins to drift very far off: this
problem is sensitive to the choice of `a`.

To build the objective function for Optim.jl, we simply call the `build_loss_objective`
function:

```@example ode
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), 
                                     Optimization.AutoForwardDiff(),
                                     maxiters=10000,verbose=false)
```

This objective function internally is calling the ODE solver to get solutions
to test against the data. The keyword arguments are passed directly to the solver.
Note that we set `maxiters` in a way that causes the differential equation solvers to
error more quickly when in bad regions of the parameter space, speeding up the
process. If the integrator stops early (due to divergence), then those parameters
are given an infinite loss, and thus this is a quick way to avoid bad parameters.
We set `verbose=false` because this divergence can get noisy. The `Optimization.AutoForwardDiff()`
is a choice of automatic differentiation, i.e. how the gradients are calculated.
For more information on this choice, see 
[the automatic differentiation choice API](https://docs.sciml.ai/Optimization/stable/API/optimization_function/#Automatic-Differentiation-Construction-Choice-Recommendations). 

!!! note 
    A good rule of thumb is to use `Optimization.AutoForwardDiff()` for less than 100
    parameters + states, and `Optimization.AutoZygote()` for more. 

Before optimizing, let's visualize our cost function by plotting it for a range
of parameter values:

```@example ode
vals = 0.0:0.1:10.0
plot(vals,[cost_function(i) for i in vals],yscale=:log10,
     xaxis = "Parameter", yaxis = "Cost", title = "1-Parameter Cost Function",
     lw = 3)
```

Here we see that there is a very well-defined minimum in our cost function at
the real parameter (because this is where the solution almost exactly fits the
dataset).

Now we can use the `BFGS` algorithm to optimize the parameter starting at `a=1.42`. We
do this by creating an optimization problem and solving that with `BFGS()`:

```@example ode
optprob = Optimization.OptimizationProblem(cost_function, [1.42])
optsol = solve(optprob, BFGS())
```

Now let's see how well the fit performed:

```@example ode
newprob = remake(prob, p = optsol.u)
newsol = solve(newprob,Tsit5())
plot(sol)
plot!(newsol)
```

Note that some of the algorithms may be sensitive to the initial condition. For more
details on using Optim.jl, see the [documentation for Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/).

### Adding Bounds Constraints

We can improve our solution by noting that the Lotka-Volterra equation requires that
the parameters are positive. Thus 
[following the Optimization.jl documentation](https://docs.sciml.ai/Optimization/stable/API/optimization_problem/)
we can add box constraints to ensure the optimizer only checks between `0.0` and `3.0`
which improves the efficiency of our algorithm. We pass the `lb` and `ub` keyword 
arguments to the `OptimizationProblem` to pass these bounds to the optimizer:

```@example ode
lower = [0.0]
upper = [3.0]
optprob = Optimization.OptimizationProblem(cost_function, [1.42], lb = lower, ub = upper)
result = solve(optprob, BFGS())
```

## Estimating Multiple Parameters Simultaneously

Lastly, we can use the same tools to estimate multiple parameters simultaneously.
Let's use the Lotka-Volterra equation with all parameters free:

```@example ode
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

```@example ode
cost_function = build_loss_objective(prob, Tsit5(), L2Loss(t,data), 
                                     Optimization.AutoForwardDiff(),
                                     maxiters=10000,verbose=false)
optprob = Optimization.OptimizationProblem(cost_function, [1.3,0.8,2.8,1.2])
result_bfgs = solve(optprob, BFGS())
```

### Alternative Cost Functions for Increased Robustness

The `build_loss_objective` with `L2Loss` is the most naive approach for parameter estimation.
There are many other 

We can also use First-Differences in L2Loss by passing the kwarg `differ_weight` which decides the contribution of the
differencing loss to the total loss.

```@example ode
cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data,differ_weight=0.3,data_weight=0.7), 
                                      Optimization.AutoForwardDiff(),
                                      maxiters=10000,verbose=false)
optprob = OptimizationProblem(cost_function, [1.3,0.8,2.8,1.2])
result_bfgs = solve(optprob, BFGS())
```

We can also use Multiple Shooting method by creating a `multiple_shooting_objective`

```@example ode
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


ms_obj = multiple_shooting_objective(ms_prob, Tsit5(), L2Loss(t, data), Optimization.AutoForwardDiff(); 
                                     discontinuity_weight = 1.0, abstol = 1e-12, reltol = 1e-12)
```

This creates the objective function that can be passed to an optimizer from which we can then get the parameter values
and the initial values of the short time periods keeping in mind the indexing. Now we mix this with
a global optimization method to improve robustness even more:

```@example ode
optsol_ms = solve(ms_obj, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters = 10_000)
```

```@example ode
optsol_ms.u[end-1:end]
```

Here as our model had 2 parameters, we look at the last 2 indexes of `result` to get our parameter values and
the rest of the values are the initial values of the shorter timespans as described in the reference section.

The objective function for Two Stage method can be created and passed to an optimizer as

```@example ode
two_stage_obj = two_stage_objective(ms_prob, t, data)
optprob = Optimization.OptimizationProblem(two_stage_obj, [1.3,0.8,2.8,1.2])
result = solve(optprob, Optim.BFGS())
```

The default kernel used in the method is `Epanechnikov` others that are available are `Uniform`,  `Triangular`,
`Quartic`, `Triweight`, `Tricube`, `Gaussian`, `Cosine`, `Logistic` and `Sigmoid`, this can be passed by the
`kernel` keyword argument. `loss_func` keyword argument can be used to pass the loss function (cost function) you want
 to use and passing a valid 
 [`adtype` argument](https://docs.sciml.ai/Optimization/stable/tutorials/intro/#Controlling-Gradient-Calculations-(Automatic-Differentiation)) enables Auto Differentiation.

## Conclusion

There are many more choices for how to improve the robustness of a parameter estimation. With all of these tools,
one likely should never do the simple "solve it with `p` and check the L2 loss", but instead use these tricks to
improve the loss landscape and increase the ability for optimizers to find the globally best parameters.