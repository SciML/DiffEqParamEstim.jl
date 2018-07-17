using DelayDiffEq, OrdinaryDiffEq, RecursiveArrayTools, Test,
      Sundials

function resrob(r,yp,y,p,tres)
    r[1]  = -p[1]*y[1] + 1.0e4*y[2]*y[3]
    r[2]  = -r[1] - 3.0e7*y[2]*y[2] - yp[2]
    r[1] -=  yp[1]
    r[3]  =  y[1] + y[2] + y[3] - 1.0
end
u0 = [1.0, 0, 0]
du0 = [-0.04, 0.04, 0.0]
p = [0.04]
prob = DAEProblem(pf,u0,du0,(0.0,100000.0),p,differential_vars=[3])
sol = solve(prob, IDA())

t = collect(range(0, stop=10, length=30))
randomized = VectorOfArray([(sol(t[i]) + .003randn(3)) for i in 1:length(t)])
data = convert(Array, randomized)

using DiffEqParamEstim, NLopt
cost_function = build_loss_objective(prob,IDA(),L2Loss(t,data),maxiter=10000,
                                          abstol=1e-8,reltol=1e-8,verbose=false)

opt = Opt(:GN_ESCH, 1)
min_objective!(opt, cost_function.cost_function2)
lower_bounds!(opt, [0.0])
upper_bounds!(opt, [1.0])
maxeval!(opt, 10000)

(minf, minx, ret) = NLopt.optimize(opt,[0.2])
@test minx[1] ≈ 0.04 atol=5e-3
