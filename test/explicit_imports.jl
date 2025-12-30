using ExplicitImports
using DiffEqParamEstim
using Test

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(DiffEqParamEstim) === nothing
    @test check_no_stale_explicit_imports(DiffEqParamEstim) === nothing
end
