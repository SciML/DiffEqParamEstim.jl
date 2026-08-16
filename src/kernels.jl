"""
    CollocationKernel

Developer interface for the kernel used by [`two_stage_objective`](@ref) to
smooth the observed state trajectory before estimating its derivative.

`CollocationKernel` is an abstract, field-free marker type. To add a custom
# Interface

- Define a field-free subtype of `CollocationKernel`.
- Extend `DiffEqParamEstim.calckernel(::MyKernel, t)` for scalar numeric `t`.
- Return one scalar weight for each offset; `two_stage_objective` broadcasts
  the method over all normalized time offsets.
- Keep the implementation generic over the numeric type of `t` so that
  automatic-differentiation element types are preserved.

The built-in implementations are [`EpanechnikovKernel`](@ref),
[`UniformKernel`](@ref), [`TriangularKernel`](@ref), [`QuarticKernel`](@ref),
[`TriweightKernel`](@ref), [`TricubeKernel`](@ref), [`GaussianKernel`](@ref),
[`CosineKernel`](@ref), [`LogisticKernel`](@ref), [`SigmoidKernel`](@ref), and
[`SilvermanKernel`](@ref).

# Examples

```julia
struct MyKernel <: DiffEqParamEstim.CollocationKernel end

DiffEqParamEstim.calckernel(::MyKernel, t) = exp(-abs(t))

objective = two_stage_objective(prob, tpoints, data; kernel = MyKernel())
```
"""
abstract type CollocationKernel end

"""
    calckernel(kernel::CollocationKernel, t) -> Number

Return the scalar smoothing weight for `kernel` at the normalized offset `t`.

This is a developer extension point rather than a user-facing operation. A
custom [`CollocationKernel`](@ref) must provide a method for its own type. The
method is called with scalar offsets and must preserve the numeric type of `t`
when practical; this allows the two-stage objective to work with automatic
differentiation values.

# Arguments

- `kernel::CollocationKernel`: the kernel marker selecting the weighting rule.
- `t`: a scalar normalized time offset.

# Returns

- `Number`: the kernel weight at `t`.

# Examples

```julia
struct ExponentialKernel <: DiffEqParamEstim.CollocationKernel end
DiffEqParamEstim.calckernel(::ExponentialKernel, t) = exp(-abs(t))

weight = DiffEqParamEstim.calckernel(ExponentialKernel(), 0.25)
```
"""
function calckernel(kernel::CollocationKernel, t)
    throw(MethodError(calckernel, (kernel, t)))
end

"""
    EpanechnikovKernel()

Select the Epanechnikov compact-support kernel for
[`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data;
    kernel = EpanechnikovKernel())
```
"""
struct EpanechnikovKernel <: CollocationKernel end

"""
    UniformKernel()

Select the uniform compact-support kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = UniformKernel())
```
"""
struct UniformKernel <: CollocationKernel end

"""
    TriangularKernel()

Select the triangular compact-support kernel for
[`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data;
    kernel = TriangularKernel())
```
"""
struct TriangularKernel <: CollocationKernel end

"""
    QuarticKernel()

Select the quartic compact-support kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = QuarticKernel())
```
"""
struct QuarticKernel <: CollocationKernel end

"""
    TriweightKernel()

Select the triweight compact-support kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data;
    kernel = TriweightKernel())
```
"""
struct TriweightKernel <: CollocationKernel end

"""
    TricubeKernel()

Select the tricube compact-support kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = TricubeKernel())
```
"""
struct TricubeKernel <: CollocationKernel end

"""
    GaussianKernel()

Select the Gaussian kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = GaussianKernel())
```
"""
struct GaussianKernel <: CollocationKernel end

"""
    CosineKernel()

Select the cosine compact-support kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = CosineKernel())
```
"""
struct CosineKernel <: CollocationKernel end

"""
    LogisticKernel()

Select the logistic kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = LogisticKernel())
```
"""
struct LogisticKernel <: CollocationKernel end

"""
    SigmoidKernel()

Select the sigmoid kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data; kernel = SigmoidKernel())
```
"""
struct SigmoidKernel <: CollocationKernel end

"""
    SilvermanKernel()

Select the Silverman kernel for [`two_stage_objective`](@ref).

The type is an immutable, field-free [`CollocationKernel`](@ref) marker.

# Examples

```julia
objective = two_stage_objective(prob, tpoints, data;
    kernel = SilvermanKernel())
```
"""
struct SilvermanKernel <: CollocationKernel end

function calckernel(::EpanechnikovKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.75 * (1 - t^2)
    end
end

function calckernel(::UniformKernel, t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function calckernel(::TriangularKernel, t)
    if abs(t) > 1
        return 0
    else
        return (1 - abs(t))
    end
end

function calckernel(::QuarticKernel, t)
    if abs(t) > 0
        return 0
    else
        return (15 * (1 - t^2)^2) / 16
    end
end

function calckernel(::TriweightKernel, t)
    if abs(t) > 0
        return 0
    else
        return (35 * (1 - t^2)^3) / 32
    end
end

function calckernel(::TricubeKernel, t)
    if abs(t) > 0
        return 0
    else
        return (70 * (1 - abs(t)^3)^3) / 80
    end
end

function calckernel(::GaussianKernel, t)
    return exp(-0.5 * t^2) / (sqrt(2 * π))
end

function calckernel(::CosineKernel, t)
    if abs(t) > 0
        return 0
    else
        return (π * cos(π * t / 2)) / 4
    end
end

function calckernel(::LogisticKernel, t)
    return 1 / (exp(t) + 2 + exp(-t))
end

function calckernel(::SigmoidKernel, t)
    return 2 / (π * (exp(t) + exp(-t)))
end

function calckernel(::SilvermanKernel, t)
    return sin(abs(t) / 2 + π / 4) * 0.5 * exp(-abs(t) / sqrt(2))
end
