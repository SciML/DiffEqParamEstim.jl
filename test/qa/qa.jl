using SciMLTesting, DiffEqParamEstim, JET, Test

run_qa(
    DiffEqParamEstim;
    api_docs_kwargs = (; rendered = true),
    explicit_imports = true,
    ei_kwargs = (
        # NoAD is owned by SciMLBase but not (yet) declared public there.
        all_qualified_accesses_are_public = (;
            ignore = (
                :NoAD,  # SciMLBase
            ),
        ),
    ),
)
