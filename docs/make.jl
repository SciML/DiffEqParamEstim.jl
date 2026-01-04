using Documenter, DiffEqParamEstim

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)
ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"

include("pages.jl")

makedocs(
    sitename = "DiffEqParamEstim.jl",
    authors = "Chris Rackauckas et al.",
    modules = [DiffEqParamEstim],
    clean = true, doctest = false, linkcheck = true,
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/DiffEqParamEstim/stable/"
    ),
    pages = pages
)

deploydocs(
    repo = "github.com/SciML/DiffEqParamEstim.jl";
    push_preview = true
)
