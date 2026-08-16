# Collocation Interface

The two-stage estimator accepts either one of the built-in kernel symbols or a
custom [`DiffEqParamEstim.CollocationKernel`](@ref) implementation. A custom
kernel is a field-free subtype that extends
[`DiffEqParamEstim.calckernel`](@ref) for scalar numeric offsets. The method
must return one scalar weight and should preserve the numeric type of its
offset so that automatic differentiation can propagate through the objective.

This is a developer interface. It is documented and versioned for packages that
extend the two-stage estimator, but it is not exported as part of the ordinary
user API.

```@docs
DiffEqParamEstim.CollocationKernel
DiffEqParamEstim.calckernel(::DiffEqParamEstim.CollocationKernel, ::Any)
DiffEqParamEstim.EpanechnikovKernel
DiffEqParamEstim.UniformKernel
DiffEqParamEstim.TriangularKernel
DiffEqParamEstim.QuarticKernel
DiffEqParamEstim.TriweightKernel
DiffEqParamEstim.TricubeKernel
DiffEqParamEstim.GaussianKernel
DiffEqParamEstim.CosineKernel
DiffEqParamEstim.LogisticKernel
DiffEqParamEstim.SigmoidKernel
DiffEqParamEstim.SilvermanKernel
```
