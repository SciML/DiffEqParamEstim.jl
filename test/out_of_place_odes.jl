using OrdinaryDiffEq

function LotkaVolterraTest_not_inplace(u,a,t)
    b,c,d = 1.0,3.0,1.0
    x,y = u[1],u[2]
    du = zeros(eltype(u),2)
    du[1] = a*x - b*x*y
    du[2] = -c*y + d*x*y
    du
end

# forward
u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob = ODEProblem(LotkaVolterraTest_not_inplace,u0,tspan,1.5)

sol = solve(prob,Tsit5())
t = collect(range(0, stop=10, length=200))
using RecursiveArrayTools # for VectorOfArray
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)

# inverse
soll = solve(prob,Tsit5())

cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
import Optim
result = Optim.optimize(cost_function, 0.0, 10.0)
