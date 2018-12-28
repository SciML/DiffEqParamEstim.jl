using OrdinaryDiffEq, ParameterizedFunctions, DiffEqBase, RecursiveArrayTools

# Here are the problems to solve

f1 = @ode_def begin
  dx = a*x - x*y
  dy = -3y + x*y
end a
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob1 = ODEProblem(f1,u0,tspan,[1.5])

f2 = @ode_def begin
  dx = a*x - x*y
  dy = -c*y + x*y
end a c
p = [1.5,3.0]
prob2 = ODEProblem(f2,u0,tspan,p)

f3 = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
prob3 = ODEProblem(f3,u0,tspan,p)

# Generate random data based off of the known solution
sol = solve(prob1,Tsit5())
t = collect(range(0, stop=10, length=200))
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)
