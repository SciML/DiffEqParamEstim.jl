# Alternative Objective Functions

These are objective functions made to be used with special fitting packages.

## LeastSquaresOptim.jl objective

`build_lsoptim_objective` builds an objective function to be used with LeastSquaresOptim.jl.

```julia
build_lsoptim_objective(prob,tspan,t,data;
                        prob_generator = (prob,p) -> remake(prob,u0=convert.(eltype(p),prob.u0),p=p),
                        kwargs...)
```

The arguments are the same as `build_loss_objective`.

## lm_fit

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
