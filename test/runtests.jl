using Pkg
using DiffEqParamEstim, Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    Pkg.activate("qa")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    include("qa/qa.jl")
else
    @time @safetestset "Explicit Imports" begin
        include("explicit_imports.jl")
    end

    @time @safetestset "Tests on ODEs" begin
        using DiffEqParamEstim
        include("tests_on_odes/test_problems.jl")
        include("tests_on_odes/l2loss_test.jl")
        include("tests_on_odes/optim_test.jl")
        include("tests_on_odes/nlopt_test.jl")
        include("tests_on_odes/two_stage_method_test.jl")
        include("tests_on_odes/blackboxoptim_test.jl")
        include("tests_on_odes/regularization_test.jl")
        include("tests_on_odes/weighted_loss_test.jl")
        include("tests_on_odes/l2_colloc_grad_test.jl")
        #include("tests_on_odes/genetic_algorithm_test.jl") # Not updated to v0.6
    end

    @time @safetestset "Multiple Shooting Objective" begin
        include("multiple_shooting_objective_test.jl")
    end
    @time @safetestset "Likelihood Loss" begin
        include("likelihood.jl")
    end
    @time @safetestset "Out-of-place ODE Tests" begin
        include("out_of_place_odes.jl")
    end
    @time @safetestset "Steady State Tests" begin
        include("steady_state_tests.jl")
    end
    @time @safetestset "DAE Tests" begin
        include("dae_tests.jl")
    end
    @time @safetestset "DDE Tests" begin
        include("dde_tests.jl")
    end
    @time @safetestset "Test on Monte" begin
        include("test_on_monte.jl")
    end
end
