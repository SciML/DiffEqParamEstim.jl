using OrdinaryDiffEq, RecursiveArrayTools

# Here are the problems to solve

f1 = function (du, u, p, t)
    du[1] = p[1] * u[1] - u[1] * u[2]
    du[2] = -3 * u[2] + u[1] * u[2]
end
u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob1 = ODEProblem(f1, u0, tspan, [1.5])

f2 = function (du, u, p, t)
    du[1] = p[1] * u[1] - u[1] * u[2]
    du[2] = -p[2] * u[2] + u[1] * u[2]
end
p = [1.5, 3.0]
prob2 = ODEProblem(f2, u0, tspan, p)

f3 = function (du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end
p = [1.5, 1.0, 3.0, 1.0]
prob3 = ODEProblem(f3, u0, tspan, p)

# Generate random data based off of the known solution
using Random
Random.seed!(123)
sol = solve(prob1, Tsit5())
t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)
