using Base.Test

### Optim Method
import Optim

println("Use Optim Brent to fit the parameter")
result = Optim.optimize(obj, 1.0, 10.0)
@test result.minimizer[1] ≈ 1.5 atol=3e-1

println("Use Optim BFGS to fit the parameter")
result = Optim.optimize(obj, [1.0], Optim.BFGS())
@test result.minimizer[1] ≈ 1.5 atol=3e-1
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)
