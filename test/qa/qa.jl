using SciMLTesting, DiffEqParamEstim, JET, Test

run_qa(
    DiffEqParamEstim;
    explicit_imports = true,
    ei_kwargs = (
        # SciMLBase/DiffEqBase solution+problem supertypes that DiffEqParamEstim
        # dispatches on / subtypes; not (yet) declared public in their owner module.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractDEProblem,        # SciMLBase
                :AbstractEnsembleSolution, # SciMLBase
                :AbstractNoTimeSolution,   # SciMLBase
                :NoAD,                     # SciMLBase
                :DECostFunction,           # DiffEqBase
            ),
        ),
    ),
)
