using OrdinaryDiffEq, RecursiveArrayTools

# Here are the problems to solve

function f1!(du, u, p, t)
    a = p[1]
    x, y = u
    du[1] = a * x - x * y
    du[2] = -3y + x * y
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5]
prob1 = ODEProblem(f1!, u0, tspan, [1.5])

function f2!(du, u, p, t)
    a, c = p
    x, y = u
    du[1] = a * x - x * y
    du[2] = -c * y + x * y
end

p = [1.5, 3.0]
prob2 = ODEProblem(f2!, u0, tspan, p)

function f3!(du, u, p, t)
    a, b, c, d = p
    x, y = u
    du[1] = a * x - b * x * y
    du[2] = -c * y + d * x * y
end

p = [1.5, 1.0, 3.0, 1.0]
prob3 = ODEProblem(f3!, u0, tspan, p)

# Generate random data based off of the known solution
sol = solve(prob1, Tsit5())
t = collect(range(0, stop = 10, length = 200))
randomized = VectorOfArray([(sol(t[i]) + 0.01randn(2)) for i in 1:length(t)])
data = convert(Array, randomized)
