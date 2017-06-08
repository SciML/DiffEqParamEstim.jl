using OrdinaryDiffEq, ParameterizedFunctions, DiffEqBase, RecursiveArrayTools

# Here are the problems to solve

f1 = @ode_def_nohes LotkaVolterraTest begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=3.0 d=1.0
u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob1 = ODEProblem(f1,u0,tspan)

f2 = @ode_def_nohes LotkaVolterraMulti begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=1.0 c=>3.0 d=1.0
prob2 = ODEProblem(f2,u0,tspan)

f3 = @ode_def_nohes LotkaVolterraAll begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a=>1.5 b=>1.0 c=>3.0 d=>1.0
prob3 = ODEProblem(f3,u0,tspan)

# Generate random data based off of the known solution
sol = solve(prob1,Tsit5())
t = collect(linspace(0,10,200))
randomized = [(sol(t[i]) + .01randn(2)) for i in 1:length(t)]
data = vecvec_to_mat(randomized)
