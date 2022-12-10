# Fitting Ensembles of ODE Models to Data

In this tutoiral we will showcase how to fit multiple models simultaniously to respective
data sources. Let's dive right in!

## Formulating the Ensemble Model

First you want to create a problem which solves multiple problems at the same time. This is
the `EnsembleProblem`. When the parameter estimation tools say it will take any DEProblem,
it really means ANY DEProblem, which includes `EnsembleProblem`.

So, let's get an `EnsembleProblem` setup that solves with 10 different initial conditions.
This looks as follows:

```@example ensemble
using DifferentialEquations, DiffEqParamEstim, Plots, Optim

# Monte Carlo Problem Set Up for solving set of ODEs with different initial conditions

# Set up Lotka-Volterra system
function pf_func(du,u,p,t)
  du[1] = p[1] * u[1] - p[2] * u[1]*u[2]
  du[2] = -3 * u[2] + u[1]*u[2]
end
p = [1.5,1.0]
prob = ODEProblem(pf_func,[1.0,1.0],(0.0,10.0),p)
```

Now for an EnsembleProblem we have to take this problem and tell it what to do N times via
the `prob_func`. So let's generate N=10 different initial conditions, and tell it to run
the same problem but with these 10 different initial conditions each time:

```@example ensemble
# Setting up to solve the problem N times (for the N different initial conditions)
N = 10;
initial_conditions = [[1.0,1.0], [1.0,1.5], [1.5,1.0], [1.5,1.5], [0.5,1.0], [1.0,0.5], [0.5,0.5], [2.0,1.0], [1.0,2.0], [2.0,2.0]]
function prob_func(prob,i,repeat)
  ODEProblem(prob.f,initial_conditions[i],prob.tspan,prob.p)
end
enprob = EnsembleProblem(prob,prob_func=prob_func)
```

We can check this does what we want by solving it:

```@example ensemble
# Check above does what we want
sim = solve(enprob,Tsit5(),trajectories=N)
plot(sim)
```

trajectories=N means "run N times", and each time it runs the problem returned by the prob_func, which is always the same problem but with the ith initial condition.

Now let's generate a dataset from that. Let's get data points at every t=0.1 using saveat,
and then convert the solution into an array.

```@example ensemble
# Generate a dataset from these runs
data_times = 0.0:0.1:10.0
sim = solve(enprob,Tsit5(),trajectories=N,saveat=data_times)
data = Array(sim)
```

Here, data[i,j,k] is the same as sim[i,j,k] which is the same as sim[k][i,j] (where sim[k]
is the kth solution). So data[i,j,k] is the jth timepoint of the ith variable in the kth
trajectory.

Now let's build a loss function. A loss function is some loss(sol) that spits out a scalar
for how far from optimal we are. In the documentation I show that we normally do loss =
L2Loss(t,data), but we can bootstrap off of this. Instead lets build an array of N loss
functions, each one with the correct piece of data.

```@example ensemble
# Building a loss function
losses = [L2Loss(data_times,data[:,:,i]) for i in 1:N]
```

So losses[i] is a function which computes the loss of a solution against the data of the ith trajectory. So to build our true loss function, we sum the losses:

```@example ensemble
loss(sim) = sum(losses[i](sim[i]) for i in 1:N)
```

As a double check, make sure that loss(sim) outputs zero (since we generated the data from sim). Now we generate data with other parameters:

```@example ensemble
prob = ODEProblem(pf_func,[1.0,1.0],(0.0,10.0),[1.2,0.8])
function prob_func(prob,i,repeat)
  ODEProblem(prob.f,initial_conditions[i],prob.tspan,prob.p)
end
enprob = EnsembleProblem(prob,prob_func=prob_func)
sim = solve(enprob,Tsit5(),trajectories=N,saveat=data_times)
loss(sim)
```

and get a non-zero loss. So we now have our problem, our data, and our loss function... we
have what we need.

Put this into build_loss_objective.

```@example ensemble
obj = build_loss_objective(enprob,Tsit5(),loss,trajectories=N,
                           saveat=data_times)
```

Notice that I added the kwargs for solve into this. They get passed to an internal solve
command, so then the loss is computed on N trajectories at data_times.

Thus we take this objective function over to any optimization package. I like to do quick
things in Optim.jl. Here, since the Lotka-Volterra equation requires positive parameters,
I use Fminbox to make sure the parameters stay positive. I start the optimization with
[1.3,0.9], and Optim spits out that the true parameters are:

```@example ensemble
lower = zeros(2)
upper = fill(2.0,2)
result = optimize(obj, lower, upper, [1.3,0.9], Fminbox(BFGS()))
```

```@example ensemble
result
```

Optim finds one but not the other parameter.

I would run a test on synthetic data for your problem before using it on real data. Maybe
play around with different optimization packages, or add regularization. You may also want
to decrease the tolerance of the ODE solvers via

```@example ensemble
obj = build_loss_objective(enprob,Tsit5(),loss,trajectories=N,
                           abstol=1e-8,reltol=1e-8,
                           saveat=data_times)
result = optimize(obj, lower, upper, [1.3,0.9], Fminbox(BFGS()))
```

```@example ensemble
result
```

if you suspect error is the problem. However, if you're having problems it's most likely
not the ODE solver tolerance and mostly because parameter inference is a very hard
optimization problem.
