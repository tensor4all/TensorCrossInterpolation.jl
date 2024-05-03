using Documenter

push!(LOAD_PATH, "../src/")
using TensorCrossInterpolation

DocMeta.setdocmeta!(TensorCrossInterpolation, :DocTestSetup, :(using TensorCrossInterpolation); recursive=true)

makedocs(;
    modules=[TensorCrossInterpolation],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    repo="https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl/blob/{commit}{path}#{line}",
    sitename="TensorCrossInterpolation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl",
        edit_link="main",
        assets=String[],
        repolink="https://gitlab.com/tensors4fields/tensorcrossinterpolation.jl"
    ),
    pages=[
        "Home" => "index.md",
        "Documentation" => "documentation.md",
        "Implementation details" => "implementation.md"
    ]
)

deploydocs(;
    repo="github.com/tensor4all/TensorCrossInterpolation.jl.git",
    devbranch="main",
)
