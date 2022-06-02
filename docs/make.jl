using Documenter, DiffEqParamEstim

makedocs(
    sitename="DiffEqParamEstim.jl",
    authors="Chris Rackauckas et al.",
    clean=true,
    doctest=false,
    modules=[DiffEqParamEstim],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://diffeqparamestim.sciml.ai/stable/"),

    pages=pages
)

deploydocs(
    repo="github.com/SciML/DiffEqParamEstim.jl";
    push_preview=true
)
