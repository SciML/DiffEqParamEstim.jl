using Optim, Random
obj = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                           maxiters=10000,verbose=false)

### Optim Method

println("Use Optim Brent to fit the parameter")
result = Optim.optimize(obj, 1.0, 10.0)
@test_broken result.minimizer[1] ≈ 1.5 atol=3e-1

println("Use Optim BFGS to fit the parameter")
result = Optim.optimize(obj, [1.0], Optim.BFGS())
@test result.minimizer[1] ≈ 1.5 atol=3e-1
#sol_optimized2 = solve(prob)
#plot!(sol_optimized2,leg=false)

cost_function2 = build_loss_objective(prob2,Tsit5(),L2Loss(t,data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function2, [1.0,2.5], Optim.BFGS())
@test_broken result_bfgs.minimizer ≈ [1.5;3.0] atol=3e-1

Random.seed!(200)
cost_function3 = build_loss_objective(prob3,Tsit5(),L2Loss(t,data),
                                      maxiters=10000,verbose=false)
result_bfgs = Optim.optimize(cost_function3, [1.3,0.8,2.8,1.2], Optim.BFGS())
@test result_bfgs.minimizer ≈ [1.5;1.0;3.0;1.0] atol=5e-1

#=
println("Use Optim NelderMead to fit the parameter")
result_neldermead = Optim.optimize(cost_function2, [1.0,2.5], Optim.NelderMead())
@test result_neldermead.minimizer ≈ [1.5;3.0] atol=3e-1

println("Use Optim SimulatedAnnealing to fit the parameter")
options = Optim.Options(iterations=100_000)
result_simulatedannealing = Optim.optimize(obj, [1.0], Optim.SimulatedAnnealing(),options)
@test result_simulatedannealing.minimizer ≈ [1.5] atol=3e-1


result_simulatedannealing = Optim.optimize(cost_function2, [1.0,2.5], Optim.SimulatedAnnealing(),options)
@test result_simulatedannealing.minimizer ≈ [1.5;3.0] atol=3e-1

println("Use Optim Particle Swarm Optimization to fit the parameter")
initial_x = [1.0]
lower = [0.0]
upper = [5.0]
n_particles = 4
options = Optim.Options(iterations=100)
result_pso = Optim.optimize(obj, initial_x, Optim.ParticleSwarm(lower, upper, n_particles), options)
@test result_pso.minimizer[1] ≈ 1.5 atol=3e-1

initial_x = [1.0,2.5]
lower = [0.0,2.0]
upper = [5.0,10.0]
n_particles = 4
options = Optim.Options(iterations=100)
result_pso = Optim.optimize(cost_function2, initial_x, Optim.ParticleSwarm(lower, upper, n_particles), options)
@test result_pso.minimizer ≈ [1.5;3.0] atol=3e-1
=#
