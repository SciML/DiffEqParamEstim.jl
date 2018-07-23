using Evolutionary

println("Use Genetic Algorithm to fit the parameter")
#         objfun: Objective fitness function
#              N: Search space dimensionality
# initPopulation: Search space dimension ranges as a vector, or initial population values as matrix,
#                 or generation function which produce individual population entities.
# populationSize: Size of the population
#  crossoverRate: The fraction of the population at the next generation, not including elite children,
#                 that is created by the crossover function.
#   mutationRate: Probability of chromosome to be mutated
#              ɛ: Positive integer specifies how many individuals in the current generation
#                 are guaranteed to survive to the next generation.
#                 Floating number specifies fraction of population.

cost_function = build_loss_objective(prob1,Tsit5(),L2Loss(t,data),
                                     maxiters=10000,verbose=false)
N = 1
result, fitness, cnt = ga(cost_function, N;
                   initPopulation = Float64[1.2],
                   populationSize = 100,
                   ɛ = 0.1,
               selection = sus,
               crossover = intermediate(0.25),
               mutation = domainrange(fill(0.5,N)))
@test result[1] ≈ 1.5 atol=3e-1


cost_function = build_loss_objective(prob2,Tsit5(),L2Loss(t,data),
                                     maxiters=10000)
N = 2
result, fitness, cnt = ga(cost_function, N;
                   initPopulation = Float64[1.2,2.8],
                   populationSize = 500,
                   ɛ = 0.1,
               selection = sus,
               crossover = intermediate(0.25),
               mutation = domainrange(fill(0.5,N)))
@test result ≈ [1.5;3.0] atol=3e-1

cost_function = build_loss_objective(prob3,Tsit5(),L2Loss(t,data),
                                     maxiters=10000)
N = 4
result, fitness, cnt = ga(cost_function, N;
                   initPopulation = Float64[1.3,0.8,2.8,1.2],
                   populationSize = 1000,
                   ɛ = 0.1,
               selection = sus,
               crossover = intermediate(0.25),
               mutation = domainrange(fill(0.5,N)))
@test result ≈ [1.5;1.0;3.0;1.0] atol=3e-1
