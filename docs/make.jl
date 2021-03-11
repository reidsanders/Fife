using Fife
using Documenter

DocMeta.setdocmeta!(Fife, :DocTestSetup, :(using Fife); recursive=true)

makedocs(;
    modules=[Fife],
    authors="Reid Sanders",
    repo="https://github.com/reidsanders/Fife.jl/blob/{commit}{path}#{line}",
    sitename="Fife.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://reidsanders.github.io/Fife.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/reidsanders/Fife.jl",
)
