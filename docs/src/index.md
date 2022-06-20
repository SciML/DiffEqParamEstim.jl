# DiffEqParamEstim.jl

DiffEqParamEstim.jl is a package for simplified parameter estimation with 
[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
While not as expansive as [DiffEqSensitivity.jl](https://sensitivity.sciml.ai/dev/)
or [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl), it's a simple and helpful
for new users who want to quickly run standard parameter estimation routines for model
callibration on not too large of models (<100 parameters or ODEs). For much larger
models and more complex setups (multiple datasets, batching, etc.) see the aforementioned
DiffEqSensitivity and DiffEqFlux.

## Installation

To use this functionality, you must install DiffEqParamEstim.jl:

```julia
]add DiffEqParamEstim
using DiffEqParamEstim
```
