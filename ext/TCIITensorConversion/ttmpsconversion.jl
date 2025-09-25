function ITensorMPS.MPS(tt::TCI.TensorTrain{T}; sites=nothing)::MPS where {T}
    N = length(tt)
    localdims = [size(t, 2) for t in tt]

    if sites === nothing
        sites = [Index(localdims[n], "n=$n") for n in 1:N]
    else
        all(localdims .== dim.(sites)) ||
            error("ranks are not consistent with dimension of sites")
    end

    linkdims = [1, TCI.linkdims(tt)..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N]

    tensors_ = [ITensor(deepcopy(tt[n]), links[n], sites[n], links[n + 1]) for n in 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return MPS(tensors_)
end

"""
    MPS(tt::TCI.AbstractTensorTrain{T}; sites=nothing)

Convert a tensor train to an `ITensorMPS.MPS`.
Arguments:
- `tt`: The tensor train to be converted.
- `siteindices`: An array of ITensor Index objects, where `sites[n]` corresponds to the index for the nth site.

If `siteindices` is left empty, a default set of indices will be generated.
"""
function ITensorMPS.MPS(tt::TCI.AbstractTensorTrain{T}; sites=nothing)::MPS where {T}
    return MPS(TCI.tensortrain(tt), sites=sites)
end

function ITensorMPS.MPO(tt::TCI.TensorTrain{T}; sites=nothing)::ITensorMPS.MPO where {T}
    N = length(tt)
    localdims = TCI.sitedims(tt)

    if sites === nothing
        sites = [
            [Index(legdim, "ell=$ell, n=$n") for (n, legdim) in enumerate(ld)]
            for (ell, ld) in enumerate(localdims)
        ]
    elseif !all(all.(
            localdimell .== dim.(siteell)
            for (localdimell, siteell) in zip(localdims, sites)
        ))
        error("ranks are not consistent with dimension of sites")
    end

    linkdims = [1, TCI.linkdims(tt)..., 1]
    links = [Index(linkdims[l + 1], "link,l=$l") for l in 0:N]

    tensors_ = [ITensor(deepcopy(tt[n]), links[n], sites[n]..., links[n + 1]) for n in 1:N]
    tensors_[1] *= onehot(links[1] => 1)
    tensors_[end] *= onehot(links[end] => 1)

    return ITensorMPS.MPO(tensors_)
end

"""
    MPO(tt::TCI.AbstractTensorTrain{T}; sites=nothing)

Convert a tensor train to an `ITensorMPS.MPO`.
Arguments:
- `tt`: The tensor train to be converted.
- `sites`: An array of arrays of ITensor Index objects, where `sites[n][m]` corresponds to the mth index attached to the nth site.

If `siteindices` is left empty, a default set of indices will be generated.
"""
function ITensorMPS.MPO(tci::TCI.AbstractTensorTrain{T}; sites=nothing)::ITensorMPS.MPO where {T}
    return ITensorMPS.MPO(TCI.tensortrain(tci), sites=sites)
end


"""
    function TensorTrain(mps::ITensorMPS.MPS)

Converts an `ITensorMPS.MPS` object into a TensorTrain. Note that this only works if the MPS has a single leg per site. Otherwise, use [`TensorCrossInterpolation.TensorTrain(mps::ITensorMPS.MPO)`](@ref).
"""
function TCI.TensorTrain(mps::ITensorMPS.MPS)
    links = ITensorMPS.linkinds(mps)
    sites = ITensors.SiteTypes.siteinds(mps)
    Tfirst = zeros(ComplexF64, 1, dim(sites[1]), dim(links[1]))
    Tfirst[1, :, :] = Array(mps[1], sites[1], links[1])
    Tlast =  zeros(ComplexF64, dim(links[end]), dim(sites[end]), 1)
    Tlast[:, :, 1] = Array(mps[end], links[end], sites[end])
    return TCI.TensorTrain{ComplexF64,3}(
        vcat(
            [Tfirst],
            [Array(mps[i], links[i-1], sites[i], links[i]) for i in 2:length(mps)-1],
            [Tlast]
        )
    )
end

"""
    function TensorTrain(mps::ITensorMPS.MPO)

Converts an `ITensorMPS.MPO` object into a [`TensorCrossInterpolation.TensorTrain`](@ref).
"""
function TCI.TensorTrain{V, N}(mpo::ITensorMPS.MPO; sites=nothing) where {N, V}
    links = ITensorMPS.linkinds(mpo)
    if sites === nothing
        sites = ITensors.SiteTypes.siteinds(mpo)
    elseif !all(issetequal.(ITensors.SiteTypes.siteinds(mpo), sites))
        error("Site indices do not correspond to the site indices of the MPO.")
    end

    Tfirst = zeros(ComplexF64, 1, dim.(sites[1])..., dim(links[1]))
    Tfirst[1, fill(Colon(), length(sites[1]) + 1)...] = Array(mpo[1], sites[1]..., links[1])

    Tlast =  zeros(ComplexF64, dim(links[end]), dim.(sites[end])..., 1)
    Tlast[fill(Colon(), length(sites[end]) + 1)..., 1] = Array(mpo[end], links[end], sites[end]...)

    return TCI.TensorTrain{V, N}(
        vcat(
            Array{V, N}[Tfirst],
            Array{V, N}[Array(mpo[i], links[i-1], sites[i]..., links[i]) for i in 2:length(mpo)-1],
            Array{V, N}[Tlast]
        )
    )
end
