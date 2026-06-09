using DiffEqParamEstim, Aqua, JET, Test

@testset "Aqua" begin
    Aqua.test_all(DiffEqParamEstim)
end

@testset "JET" begin
    JET.test_package(DiffEqParamEstim; target_defined_modules = true)
end
