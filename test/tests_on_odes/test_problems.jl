using OrdinaryDiffEq, ParameterizedFunctions, DiffEqBase, RecursiveArrayTools

# Here are the problems to solve

f1 = @ode_def LotkaVolterraTest begin
  dx = a*x - x*y
  dy = -3y + x*y
end a
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob1 = ODEProblem(f1,u0,tspan,[1.5])

f2 = @ode_def LotkaVolterraMulti begin
  dx = a*x - x*y
  dy = -c*y + x*y
end a c
p = [1.5,3.0]
prob2 = ODEProblem(f2,u0,tspan,p)

f3 = @ode_def_nohes LotkaVolterraAll begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
prob3 = ODEProblem(f3,u0,tspan,p)

# Generate random data based off of the known solution

t = collect(linspace(0,10,200))
sol1 = solve(prob1,Tsit5())
randomized = VectorOfArray([(sol1(t[i]) + .01randn(2)) for i in 1:length(t)])
data1 = convert(Array,randomized)

sol2 = solve(prob2,Tsit5())
randomized = VectorOfArray([(sol2(t[i]) + .01randn(2)) for i in 1:length(t)])
data2 = convert(Array,randomized)

sol3 = solve(prob3,Tsit5())
randomized = VectorOfArray([(sol3(t[i]) + .01randn(2)) for i in 1:length(t)])
data3 = convert(Array,randomized)