using SciMLTesting, DiffEqParamEstim, JET, Test

run_qa(
    DiffEqParamEstim;
    explicit_imports = true,
    ei_kwargs = (
        # SciMLBase/DiffEqBase solution+problem supertypes and helpers that
        # DiffEqParamEstim dispatches on / subtypes; not (yet) declared public in
        # their owner module. They go public as those base libraries release.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEProblem,        # SciMLBase
                :AbstractEnsembleSolution, # SciMLBase
                :AbstractNoTimeSolution,   # SciMLBase
                :AbstractSciMLProblem,     # SciMLBase
                :AbstractSciMLSolution,    # SciMLBase
                :NoAD,                     # SciMLBase
                :build_solution,           # SciMLBase
                :successful_retcode,       # SciMLBase (flagged on Julia 1.10 only)
                :Success,                  # SciMLBase.ReturnCode (flagged on Julia 1.10 only)
                :DECostFunction,           # DiffEqBase
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                :solve,         # CommonSolve canonical entry point, not declared public there
                :loglikelihood, # StatsAPI (flagged on Julia 1.10 only)
            ),
        ),
    ),
)
