################################################################################

# Setup

# Note: This version is against the true solution, so adaptive timestepping is
# used. This means that the option which is set to get the correct timepoints
# is saveat, not tstops!

using DifferentialEquations, RecursiveArrayTools
using NLopt
using BlackBoxOptim

Xiang2015Bounds = Tuple{Float64, Float64}[(9, 11), (20, 30), (2, 3)] # for local optimizations
xlow_bounds = [9.0,20.0,2.0]
xhigh_bounds = [11.0,30.0,3.0]
LooserBounds = Tuple{Float64, Float64}[(0, 22), (0, 60), (0, 6)] # for global optimization
GloIniPar = [0.0, 0.5, 0.1] # for global optimizations
LocIniPar = [9.0, 20.0, 2.0] # for local optimization

g1 = @ode_def_nohes LorenzExample begin
  dx = σ*(y-x)
  dy = x*(ρ-z) - y
  dz = x*y - β*z
end σ=>10.0 ρ=>28.0 β=>2.6666

r0 = [0.1;0.0;0.0]
tspan = (0.0,30.0)
prob = ODEProblem(g1,r0,tspan)
tspan2 = (0.0,3.0)
prob_short = ODEProblem(g1,r0,tspan2)

dt = 30.0/3000
tf = 30.0
tinterval = 0:dt:tf
t  = collect(tinterval)

h = 0.01
M = 300
tstart = 0.0
tstop = tstart + M * h
tinterval_short = 0:h:tstop
t_short = collect(tinterval_short)

# Generate Data
data_sol_short = solve(prob_short,Vern7(),saveat=t_short,reltol=1e-12,abstol=1e-12)
data_short = convert(Array,data_sol_short)
data_sol = solve(prob,Vern7(),saveat=t,reltol=1e-12,abstol=1e-12)
data = convert(Array,data_sol)

# Plot the data

# using Plots
# plot(data_sol_short) # the short solution
# plot(data_sol) # the longer solution

################################################################################

# Using BlackBoxOptim

# Note: Euler uses tstops to hit the estimation timepoints exactly since it's not adaptive
obj_short = build_loss_objective(prob_short,Euler(),L2Loss(t_short,data_short),tstops=t_short)
res1 = bboptimize(obj_short;SearchRange = Xiang2015Bounds, MaxSteps = 11e3)
# Euler could not recover the correct results since its error is too high!

obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short))
res1 = bboptimize(obj_short;SearchRange = Xiang2015Bounds, MaxSteps = 11e3)
# Tolernace is still too high to get close enough

obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),reltol=1e-9)
res1 = bboptimize(obj_short;SearchRange = Xiang2015Bounds, MaxSteps = 11e3)
# With the tolerance lower, it achieves the correct solution in 4.5 seconds.

obj_short = build_loss_objective(prob_short,Vern7(),L2Loss(t_short,data_short),reltol=1e-12,abstol=1e-12)
res1 = bboptimize(obj_short;SearchRange = Xiang2015Bounds, MaxSteps = 11e3)
# But too much accuracy in the numerical solution of the ODE actually leads to
# slower convergence, since each step takes longer!

# Fastest BlackBoxOptim: 4.5 seconds

################################################################################

# using NLopt
obj_short = build_loss_objective(prob_short,Tsit5(),L2Loss(t_short,data_short),reltol=1e-14)

opt = Opt(:GN_ORIG_DIRECT_L, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

opt = Opt(:GN_CRS2_LM, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

opt = Opt(:GN_ISRES, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

opt = Opt(:LN_BOBYQA, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)
# This one took 0.04 seconds! Wow!

opt = Opt(:LN_NELDERMEAD, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

opt = Opt(:LD_SLSQP, 3)
lower_bounds!(opt,LocIniPar)
upper_bounds!(opt,[11.0,30.0,3.0])
min_objective!(opt, obj_short)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

################################################################################

# Longer version

obj = build_loss_objective(prob,Euler(),L2Loss(t,data),tstops = t)
# res1 = bboptimize(obj;SearchRange = Xiang2015Bounds, MaxSteps = 8e3)
# Once again, Euler fails to convergence its error is too high

obj = build_loss_objective(prob,Vern7(),L2Loss(t,data),reltol=1e-14)
# res1 = bboptimize(obj;SearchRange = Xiang2015Bounds, MaxSteps = 8e3)
# BB with Tsit5 converges just fine in 14.5 seconds

opt = Opt(:GN_ORIG_DIRECT_L, 3)
lower_bounds!(opt,zeros(3))
upper_bounds!(opt,[22.0,60.0,6.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,GloIniPar)

opt = Opt(:GN_CRS2_LM, 3)
lower_bounds!(opt,zeros(3))
upper_bounds!(opt,[22.0,60.0,6.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, Int(1e5))
@time (minf,minx,ret) = NLopt.optimize(opt,GloIniPar)

opt = Opt(:GN_ISRES, 3)
lower_bounds!(opt,zeros(3))
upper_bounds!(opt,[22.0,60.0,6.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, Int(1e6))
@time (minf,minx,ret) = NLopt.optimize(opt,GloIniPar)

opt = Opt(:GN_ESCH, 3)
lower_bounds!(opt,zeros(3))
upper_bounds!(opt,[22.0,60.0,6.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, Int(1e5))
@time (minf,minx,ret) = NLopt.optimize(opt,GloIniPar)

################################################################

# Local Optimizers fail on the hard problem, as expected
LocIniPar = [11.5312, 26.0192, 2.79983]
opt = Opt(:LN_BOBYQA, 3)
lower_bounds!(opt,xlow_bounds)
upper_bounds!(opt,[13.0,30.0,3.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, Int(1e5))
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)
# Converges in 0.05 seconds!

opt = Opt(:LN_NELDERMEAD, 3)
lower_bounds!(opt,xlow_bounds)
upper_bounds!(opt,[13.0,30.0,3.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)

opt = Opt(:LD_SLSQP, 3)
lower_bounds!(opt,xlow_bounds)
upper_bounds!(opt,[13.0,30.0,3.0])
min_objective!(opt, obj)
xtol_rel!(opt,1e-12)
maxeval!(opt, 10000)
@time (minf,minx,ret) = NLopt.optimize(opt,LocIniPar)


################################################################################

# Conclusion:

# 1) When doing these experiments, making sure the "data" is correct is key.
#    The lorenz_test.jl results are all skewed because the actual Euler solution
#    that made the data is skewed. But, since it's a fixed timestep, it was able
#    to find the parameters that made the data, given that it's hitting all the
#    same points. That's not the case here. Because there's adaptive timestepping,
#    we're getting rid of the ability to "overfit", i.e. simply match the data
#    we generated. Instead, we have to hit the same solution, albeit at different
#    stepping points, making it less prone to the bias.
# 2) When trying to hit the real data, having a low enough tolerance on the
#    numerical solution is key. If the numerical solution is too rough, then
#    we can never actually hone in on the true parameters since even with the
#    true parameters we will erroneously induce numerical error. Maybe this
#    could be adaptive?
# 3) Too low of a tolerance in the numerical solution is unnecessary and will
#    slow down the convergence of the estimation.
# 4) The estimation method makes a huge difference in the timings. Here, BBO
#    doesn't do very well. Some of the methods in NLopt, like :LN_BOBYQA, solve
#    the problem in <0.05 seconds. This is an algorithm that can scale.
# 5) Fitting shorter timespans is easier... maybe this could be some kind of
#    adaptivity trick as well?
