using DiffEqParamEstim
using SciMLBase
using Test

@testset "Precompile workload API" begin
    rhs = function (du, u, p, t)
        du[1] = p[1] * u[1]
        return du[2] = -p[1] * u[2]
    end
    problem = ODEProblem(rhs, [1.0, 1.0], (0.0, 1.0), [0.5])
    times = collect(range(0.0, 1.0; length = 5))
    data = reshape(collect(range(1.0, 2.0; length = 10)), 2, :)

    objective = two_stage_objective(problem, times, data)
    @test objective isa SciMLBase.OptimizationFunction
    @test L2Loss(times, data) isa L2Loss
end
