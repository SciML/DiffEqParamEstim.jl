using DiffEqParamEstim, Base.Test

tic()
@time @testset "Tests on ODEs" begin
  include("tests_on_odes/test_problems.jl")
  include("tests_on_odes/optim_test.jl")
  include("tests_on_odes/lm_test.jl")
  include("tests_on_odes/lsoptim_test.jl")
  include("tests_on_odes/nlopt_test.jl")
  include("tests_on_odes/two_stage_method_test.jl")
  include("tests_on_odes/blackboxoptim_test.jl")
  include("tests_on_odes/genetic_algorithm_test.jl")
end

@time @testset "ParameterizedFunction Type" begin include("parameterized_function_type.jl") end
@time @testset "Test on Monte" begin include("test_on_monte.jl") end
toc()
