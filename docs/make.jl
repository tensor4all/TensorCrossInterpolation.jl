using TensorCrossInterpolation
using ITensors
using ITensorMPS
using Documenter

preamble = quote
    using TensorCrossInterpolation
    using ITensors
    using ITensorMPS
end

DocMeta.setdocmeta!(TensorCrossInterpolation, :DocTestSetup, preamble; recursive=true)

makedocs(;
    modules=[
        TensorCrossInterpolation,
        Base.get_extension(TensorCrossInterpolation, :TCIITensorConversion)
    ],
    authors="Ritter.Marc <Ritter.Marc@physik.uni-muenchen.de> and contributors",
    sitename="TensorCrossInterpolation.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/TensorCrossInterpolation.jl",
        edit_link="main",
        assets=String[]),
    pages=[
        "Home" => "index.md",
        "Documentation" => "documentation.md",
        "Extensions" => "extensions.md",
    ]
)

deploydocs(;
    repo="github.com/tensor4all/TensorCrossInterpolation.jl.git",
    devbranch="main",
)
