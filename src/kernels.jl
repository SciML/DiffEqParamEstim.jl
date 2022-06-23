# Kernel definition is taken from here: https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use

abstract type CollocationKernel end
struct EpanechnikovKernel <: CollocationKernel end
struct UniformKernel <: CollocationKernel end
struct TriangularKernel <: CollocationKernel end
struct QuarticKernel <: CollocationKernel end
struct TriweightKernel <: CollocationKernel end
struct TricubeKernel <: CollocationKernel end
struct GaussianKernel <: CollocationKernel end
struct CosineKernel <: CollocationKernel end
struct LogisticKernel <: CollocationKernel end
struct SigmoidKernel <: CollocationKernel end
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
    exp(-0.5 * t^2) / (sqrt(2 * π))
end

function calckernel(::CosineKernel, t)
    if abs(t) > 0
        return 0
    else
        return (π * cos(π * t / 2)) / 4
    end
end

function calckernel(::LogisticKernel, t)
    1 / (exp(t) + 2 + exp(-t))
end

function calckernel(::SigmoidKernel, t)
    2 / (π * (exp(t) + exp(-t)))
end

function calckernel(::SilvermanKernel, t)
    sin(abs(t) / 2 + π / 4) * 0.5 * exp(-abs(t) / sqrt(2))
end
