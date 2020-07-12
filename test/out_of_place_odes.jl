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

# two-stage OOP regression test

function ff(u, p, t)
    du .= p .* u
end
function ff(u, p, t)
    p .* u
end
rc=62
ps = repeat([-0.001], rc)
tspan = (7.0, 84.0)
u0 = 3.4 .+ ones(rc)
t = collect(range(minimum(tspan), stop=maximum(tspan), length=157))
prob = ODEProblem(ff, u0, tspan, ps)
prob_oop = ODEProblem{false}(ff, u0, tspan, ps)
data = Array(solve(prob, Tsit5(), saveat = t))
ptest = ones(rc)

obj_ts = two_stage_method(prob, t, data; kernel=:Sigmoid)
@test obj_ts(ptest) ≈ 418.3400017500223
obj_ts = two_stage_method(prob_oop, t, data; kernel=:Sigmoid)
@test obj_ts(ptest) ≈ 418.3400017500223
