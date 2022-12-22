# Global Optimization via NLopt

The `build_loss_objective` function builds an objective function which is able
to be used with MathOptInterface-associated solvers. This includes packages like
IPOPT, NLopt, MOSEK, etc. Building off of the previous example, we can build a
cost function for the single parameter optimization problem like:

```@example global_optimization
using DifferentialEquations, Plots, DiffEqParamEstim, Optimization, OptimizationMOI, OptimizationNLopt, NLopt

function f(du,u,p,t)
  du[1] = p[1]*u[1] - u[1]*u[2]
  du[2] = -3*u[2] + u[1]*u[2]
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5]
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob,Tsit5())

t = collect(range(0,stop=10,length=200))
randomized = VectorOfArray([(sol(t[i]) + .01randn(2)) for i in 1:length(t)])
data = convert(Array,randomized)

obj = build_loss_objective(prob,Tsit5(),L2Loss(t,data),Optimization.AutoForwardDiff())
```

You can either use the NLopt package directly or through either the OptimizationNLopt or OptimizationMOI which provides interface for all MathOptInterface compatible non-linear solvers.

We can now use this `obj` as the objective function with MathProgBase solvers.
For our example, we will use NLopt. To use the local derivative-free
Constrained Optimization BY Linear Approximations algorithm, we can simply do:

```@example global_optimization
opt = Opt(:LN_COBYLA, 1)
optprob = Optimization.OptimizationProblem(obj, [1.3])
res = solve(optprob, opt)
```

For a modified evolutionary algorithm, we can use:

```@example global_optimization
opt = Opt(:GN_ESCH, 1)
lower_bounds!(opt,[0.0])
upper_bounds!(opt,[5.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 100000)
res = solve(optprob,opt)
```

We can even use things like the Improved Stochastic Ranking Evolution Strategy
(and add constraints if needed). Let's use this through OptimizationMOI:

```@example global_optimization
optprob = Optimization.OptimizationProblem(obj, [0.2], lb = [-1.0], ub = [5.0])
res = solve(optprob, OptimizationMOI.MOI.OptimizerWithAttributes(NLopt.Optimizer, "algorithm" => :GN_ISRES, "xtol_rel" => 1e-3, "maxeval" => 10000))
```

which is very robust to the initial condition. We can also directly use the NLopt interface as below. The fastest result comes from the
following algorithm choice:

```example global_optimization
opt = Opt(:LN_BOBYQA, 1)
min_objective!(opt, obj)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
```

For more information, see the NLopt documentation for more details. And give IPOPT
or MOSEK a try!
