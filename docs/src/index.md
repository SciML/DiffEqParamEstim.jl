# DiffEqParamEstim.jl: Parameter Estimation for Differential Equations

DiffEqParamEstim.jl is a package for simplified parameter estimation with 
[DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
While not as expansive as [SciMLSensitivity.jl](https://sensitivity.sciml.ai/dev/), 
it's a simple and helpful for new users who want to quickly run standard parameter 
estimation routines for model callibration on not too large of models (<100 parameters 
or ODEs). 

!!! note

    For much larger models and more complex setups (multiple datasets, batching, etc.) see 
    [SciMLSensitivity](https://sensitivity.sciml.ai/dev/).


## Installation

To install DiffEqParamEstim.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("DiffEqParamEstim")
```

## Contributing

- Please refer to the
  [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac/blob/master/README.md)
  for guidance on PRs, issues, and other matters relating to contributing to SciML.
- See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
- There are a few community forums:
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Slack](https://julialang.org/slack/)
    - The #diffeq-bridged and #sciml-bridged channels in the
      [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
    - On the [Julia Discourse forums](https://discourse.julialang.org)
    - See also [SciML Community page](https://sciml.ai/community/)
