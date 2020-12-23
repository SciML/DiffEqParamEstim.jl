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
        "Tutorials" => Any[
          "tutorials/ODE_inference.md",
          "tutorials/global_optimization.md",
          "tutorials/jump.md",
          "tutorials/generalized_likelihood.md",
          "tutorials/stochastic_evaluations.md"
        ],
        "Methods" => Any[
          "methods/recommended_methods.md",
          "methods/optimization_based_methods.md",
          "methods/collocation_loss.md",
          "methods/alternative_objectives.md"
        ],
    ]
)

deploydocs(
    repo="github.com/SciML/DiffEqParamEstim.jl";
    push_preview=true
)
