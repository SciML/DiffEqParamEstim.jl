using DiffEqParamEstim, Aqua, JET, Test

@testset "Aqua" begin
    # stale_deps and deps_compat disabled: genuine findings tracked in
    # https://github.com/SciML/DiffEqParamEstim.jl/issues/306
    Aqua.test_all(DiffEqParamEstim; stale_deps = false, deps_compat = false)
    @test_broken false  # Aqua stale_deps: Calculus is a stale dep — tracked in https://github.com/SciML/DiffEqParamEstim.jl/issues/306
    @test_broken false  # Aqua deps_compat: LinearAlgebra dep + Pkg extra missing compat entries — tracked in https://github.com/SciML/DiffEqParamEstim.jl/issues/306
end

@testset "JET" begin
    JET.test_package(DiffEqParamEstim; target_defined_modules = true)
end
