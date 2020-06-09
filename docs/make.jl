using FluxModels
using Documenter

makedocs(;
    modules=[FluxModels],
    authors="Kyle Daruwalla",
    repo="https://github.com/darsnack/FluxModels.jl/blob/{commit}{path}#L{line}",
    sitename="FluxModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://darsnack.github.io/FluxModels.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/darsnack/FluxModels.jl",
)
