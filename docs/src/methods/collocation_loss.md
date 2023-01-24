# Two Stage method (Non-Parametric Collocation)

The two-stage method is a collocation method for estimating parameters without
requiring repeated solving of the differential equation. It does so by
determining a smoothed estimated trajectory of the data (local quadratic polynomial
fit by least squares) and optimizing
the derivative function and the data's timepoints to match the derivatives
of the smoothed trajectory. This method has less accuracy than other methods
but is much faster, and is a good method to try first to get in the general
“good parameter” region, to then finish using one of the other methods.

```julia
function two_stage_objective(prob::DEProblem, tpoints, data, adtype = SciMLBase.NoAD(), ;
                             kernel = :Epanechnikov,
                             loss_func = L2DistLoss)
end
```
