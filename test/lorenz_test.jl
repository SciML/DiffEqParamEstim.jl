using RecursiveArrayTools, ParameterizedFunctions, Zygote
using OptimizationNLopt, DiffEqParamEstim, OptimizationBBO, OptimizationBBO.BlackBoxOptim
Xiang2015Bounds = Tuple{Float64, Float64}[(9, 11), (20, 30), (
    2, 3)]

g1 = @ode_def LorenzExample begin
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
end σ ρ β

r0 = [0.1; 0.0; 0.0]
tspan = (0.0, 4.0)
p = [10.0, 28.0, 2.66666]
prob = ODEProblem(g1, r0, tspan, p)
tspan2 = (0.0, 3.0)
prob_short = ODEProblem(g1, r0, tspan2, p)

dt = 0.001
tf = 4.0
tinterval = 0:dt:tf
t = collect(tinterval)

h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

# Generate Data
data_short = convert(Array, solve(prob_short, Euler(), tstops = t_short))
data = convert(Array, solve(prob, Euler(), tstops = t))

# Use BlackBoxOptim
obj_short = build_loss_objective(prob_short, Euler(), L2Loss(t_short, data_short),
    tstops = t_short, dense = false)
res1 = bboptimize((x) -> obj_short(x, nothing); SearchRange = Xiang2015Bounds,
    MaxSteps = 11e3)
optprob = Optimization.OptimizationProblem(obj_short, [9.0, 20.0, 2.0];
    lb = [9.0, 20.0, 2.0],
    ub = [11.0, 30.0, 3.0])
res2 = solve(optprob, BBO_adaptive_de_rand_1_bin_radiuslimited())

# Use NLopt
obj_short = build_loss_objective(prob_short, Euler(), L2Loss(t_short, data_short),
    Optimization.AutoForwardDiff(), tstops = t_short,
    dense = false)
opt = Opt(:GN_ORIG_DIRECT_L, 3)
optprob = Optimization.OptimizationProblem(obj_short, [9.0, 20.0, 2.0];
    lb = [9.0, 20.0, 2.0],
    ub = [11.0, 30.0, 3.0])
@time res = solve(optprob, opt)

# opt = Opt(:GN_CRS2_LM, 3)
# @time res = solve(optprob, opt)

# opt = Opt(:GN_ISRES, 3)
# @time res = solve(optprob, opt)

opt = Opt(:LN_BOBYQA, 3)
@time res = solve(optprob, opt)

opt = Opt(:LN_NELDERMEAD, 3)
@time res = solve(optprob, opt)

opt = Opt(:LD_SLSQP, 3)
@time res = solve(optprob, opt)

#####################
# Fails to converge
#####################

opt = Opt(:GN_ESCH, 3)
res = solve(optprob, opt)

opt = Opt(:LN_COBYLA, 3)
res = solve(optprob, opt)

opt = Opt(:LN_NEWUOA_BOUND, 3)
res = solve(optprob, opt)

opt = Opt(:LN_PRAXIS, 3)
res = solve(optprob, opt)

opt = Opt(:LN_SBPLX, 3)
res = solve(optprob, opt)

# opt = Opt(:LD_MMA, 3)
# res = solve(optprob, opt)

opt = Opt(:LD_LBFGS, 3)
res = solve(optprob, opt)

opt = Opt(:LD_TNEWTON_PRECOND_RESTART, 3)
res = solve(optprob, opt)

opt = Opt(:LD_VAR2, 3)
res = solve(optprob, opt)

########

#### Now let's solve the longer version

obj = build_loss_objective(prob, Euler(), L2Loss(t, data), Optimization.AutoZygote(),
    tstops = t, dense = false)
# res1 = bboptimize(obj;SearchRange = Xiang2015Bounds, MaxSteps = 8e3)

opt = Opt(:GN_ORIG_DIRECT_L, 3)
res = solve(optprob, opt)

# opt = Opt(:GN_CRS2_LM, 3)
# res = solve(optprob, opt)

# opt = Opt(:GN_ISRES, 3)
# res = solve(optprob, opt)

opt = Opt(:LN_BOBYQA, 3)
res = solve(optprob, opt)

opt = Opt(:LN_NELDERMEAD, 3)
res = solve(optprob, opt)

opt = Opt(:LD_SLSQP, 3)
res = solve(optprob, opt)
