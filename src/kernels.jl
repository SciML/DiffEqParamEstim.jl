# Kernel definition is taken from here: https://en.wikipedia.org/wiki/Kernel_(statistics)#Kernel_functions_in_common_use

export Epanechnikov_kernel, Uniform_kernel, Triangular_kernel, Quartic_Kernel, Triweight_Kernel, Tricube_Kernel, Gaussian_Kernel, Cosine_Kernel, Logistic_Kernel, Sigmoid_Kernel, Silverman_Kernel

function Epanechnikov_kernel(t)
    if abs(t) > 1
        return 0
    else
        return 0.75*(1-t^2)
    end
end

function Uniform_kernel(t)
    if abs(t) > 1
        return 0
    else
        return 0.5
    end
end

function Triangular_kernel(t)
    if abs(t) > 1
        return 0
    else
        return (1-abs(t))
    end
end

function Quartic_Kernel(t)
  if abs(t)>0
    return 0
  else
    return (15*(1-t^2)^2)/16
  end
end

function Triweight_Kernel(t)
  if abs(t)>0
    return 0
  else
    return (35*(1-t^2)^3)/32
  end
end

function Tricube_Kernel(t)
  if abs(t)>0
    return 0
  else
    return (70*(1-abs(t)^3)^3)/80
  end
end

function Gaussian_Kernel(t)
  exp(-0.5*t^2)/(sqrt(2*π))
end

function Cosine_Kernel(t)
  if abs(t)>0
    return 0
  else
    return (π*cos(π*t/2))/4
  end
end

function Logistic_Kernel(t)
  1/(exp(t)+2+exp(-t))
end

function Sigmoid_Kernel(t)
  2/(π*(exp(t)+exp(-t)))
end

function Silverman_Kernel(t)
  sin(abs(t)/2+π/4)*0.5*exp(-abs(t)/sqrt(2))
end
