using OrdinaryDiffEq, SteadyStateDiffEq, DiffEqParamEstim, Optim, Test

function f(du,u,p,t)
    α = p[1]
    du[1] = 2 -  α*u[1]
    du[2] = u[1] - 4u[2]
end

p = [2.0]
u0 = zeros(2)
s_prob = SteadyStateProblem(f,u0,p)
s_sol = solve(s_prob,SSRootfind())
s_sol = solve(s_prob,DynamicSS(Tsit5(),abstol=1e-4,reltol=1e-3))


# true data is 1.00, 0.25
data = [1.05, 0.23]
obj = build_loss_objective(s_prob,SSRootfind(),L2Loss([Inf],data),
                           maxiters = Int(1e8),
                           abstol = 1e-10, reltol = 1e-10, verbose=true)
result = Optim.optimize(obj, [2.0], Optim.BFGS())
@test result.minimizer[1] ≈ 2.0 atol=2e-1
