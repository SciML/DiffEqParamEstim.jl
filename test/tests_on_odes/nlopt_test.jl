using OptimizationNLopt, Zygote

println("Use NLOpt to fit the parameter")

obj = build_loss_objective(
    prob1, Tsit5(), L2Loss(t, data), Optimization.AutoZygote(),
    maxiters = 10000, verbose = false
)

opt = Opt(:LN_COBYLA, 1)
optprob = OptimizationNLopt.OptimizationProblem(obj, [1.4])
res = solve(optprob, opt)
@test res.u[1] ≈ 1.5 atol = 1.0e-3

opt = Opt(:GN_ESCH, 1)
lower_bounds!(opt, [1.0])
upper_bounds!(opt, [3.0])
xtol_rel!(opt, 1.0e-3)
maxeval!(opt, 10000)
res = solve(optprob, opt)
@test res.u[1] ≈ 1.5 atol = 1.0e-1

opt = Opt(:GN_ISRES, 1)
lower_bounds!(opt, [1.0])
upper_bounds!(opt, [3.0])
xtol_rel!(opt, 1.0e-4)
maxeval!(opt, 100 - 000)
res = solve(optprob, opt)
@test res.u[1] ≈ 1.5 atol = 1.0e-1

# test differentiation

obj = build_loss_objective(
    prob1, Tsit5(), L2Loss(t, data), Optimization.AutoForwardDiff();
    maxiters = 10000
) #zygote behaves weirdly here
opt = Opt(:LD_MMA, 1)
xtol_rel!(opt, 1.0e-3)
maxeval!(opt, 10000)
optprob = OptimizationNLopt.OptimizationProblem(obj, [1.3])
res = solve(optprob, opt)
@test res.u[1] ≈ 1.5 atol = 1.0e-1
