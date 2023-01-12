# Recommended Methods

The recommended method is to use `build_loss_objective` with the optimizer
of your choice. This method can thus be paired with global optimizers
from packages like BlackBoxOptim.jl or NLopt.jl which can be much less prone to
finding local minima than local optimization methods. Also, it allows the user
to define the cost function in the way they choose as a function
`loss(sol)`. This package can thus fit using any cost function on the solution,
making it applicable to fitting non-temporal data and other types of
problems. Also, `build_loss_objective` works for all the `DEProblem`
types, allowing it to optimize parameters on ODEs, SDEs, DDEs, DAEs,
etc.

However, this method requires repeated solution of the differential equation.
If the data is temporal data, the most efficient method is the
`two_stage_objective` which does not require repeated solutions but is not as
accurate. Usage of the `two_stage_objective` should have a post-processing step
which refines using a method like `build_loss_objective`.
