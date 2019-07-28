using DiffEqParamEstim, Test

@time @testset "Tests on ODEs" begin
  include("tests_on_odes/test_problems.jl")
  include("tests_on_odes/l2loss_test.jl")
  include("tests_on_odes/optim_test.jl")
  include("tests_on_odes/lm_test.jl")
  include("tests_on_odes/lsoptim_test.jl")
  include("tests_on_odes/nlopt_test.jl")
  include("tests_on_odes/two_stage_method_test.jl")
  include("tests_on_odes/regularization_test.jl")
  include("tests_on_odes/blackboxoptim_test.jl")
  include("tests_on_odes/weighted_loss_test.jl")
  include("tests_on_odes/l2_colloc_grad_test.jl")
  #include("tests_on_odes/genetic_algorithm_test.jl") # Not updated to v0.6
end

@time @testset "Multiple Shooting Objective" begin include("multiple_shooting_objective_test.jl") end
@time @testset "Likelihood Loss" begin include("likelihood.jl") end
@time @testset "Out-of-place ODE Tests" begin include("out_of_place_odes.jl") end
@time @testset "Steady State Tests" begin include("steady_state_tests.jl") end
@time @testset "DDE Tests" begin include("dde_tests.jl") end
@time @testset "Test on Monte" begin include("test_on_monte.jl") end
