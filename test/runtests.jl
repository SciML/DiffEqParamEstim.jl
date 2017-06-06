using DiffEqParamEstim, Base.Test

tic()
@time @testset "Tests on ODEs" begin include("tests_on_odes.jl") end
@time @testset "ParameterizedFunction Type" begin include("parameterized_function_type.jl") end
@time @testset "Test on Monte" begin include("test_on_monte.jl") end
toc()
