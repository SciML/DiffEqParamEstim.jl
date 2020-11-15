using Documenter, DiffEqParamEstim

makedocs(
    sitename="DiffEqParamEstim.jl",
    authors="Chris Rackauckas et al.",
    clean=true,
    doctest=false,
    modules=[DiffEqParamEstim],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://diffeqparamestim.sciml.ai/stable/"),

    pages=[
        "DiffEqParamEstim.jl: Parameter Estimation for Differential Equations" => "index.md",
        "Methods" => Any[
          "methods/recommended_methods.md",
          "methods/optimization_based_methods.md"
        ],
        "Examples" => Any[
          "examples/ODE_inference.md"
        ],
        "Tutorials" => Any[
          "tutorials/stochastic_evaluations.md"
        ],
    ]
)

deploydocs(
    repo="github.com/SciML/DiffEqParamEstim.jl";
    push_preview=true
)
