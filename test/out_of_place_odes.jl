using DifferentialEquations

function LotkaVolterraTest_not_inplace(t,u,a)
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
prob = ODEProblem((t,u)->LotkaVolterraTest_not_inplace(t,u,1.5),u0,tspan)

sol = solve(prob,Tsit5())
t = collect(linspace(0,10,200))
using RecursiveArrayTools # for VectorOfArray
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)

# inverse
pf = ParameterizedFunction(LotkaVolterraTest_not_inplace, 1.5)
prob = ODEProblem(pf,u0,tspan)
soll = solve(prob,Tsit5())

cost_function = build_loss_objective(prob,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
import Optim
result = Optim.optimize(cost_function, 0.0, 10.0)
