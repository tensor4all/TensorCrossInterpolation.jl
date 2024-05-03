using TensorCrossInterpolation
using Documenter

DocMeta.setdocmeta!(TensorCrossInterpolation, :DocTestSetup, :(using TensorCrossInterpolation); recursive=true)

makedocs(;
    modules=[TensorCrossInterpolation],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    sitename="TensorCrossInterpolation.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/TensorCrossInterpolation.jl",
        edit_link="main",
        assets=String[]),
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
