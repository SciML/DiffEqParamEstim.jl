using DiffEqParamEstim, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(DiffEqParamEstim)
    Aqua.test_ambiguities(DiffEqParamEstim, recursive = false)
    Aqua.test_deps_compat(DiffEqParamEstim)
    Aqua.test_piracies(DiffEqParamEstim)
    Aqua.test_project_extras(DiffEqParamEstim)
    Aqua.test_stale_deps(DiffEqParamEstim)
    Aqua.test_unbound_args(DiffEqParamEstim)
    Aqua.test_undefined_exports(DiffEqParamEstim)
end
