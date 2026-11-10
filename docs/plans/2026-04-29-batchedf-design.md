# `batchedf` Batch Evaluation Design

## Goal

Replace the inheritance-only batch evaluation API with a function-based API that accepts an explicit batch evaluator, while preserving existing `BatchEvaluator` behavior for compatibility.

## Public API

Add a `batchedf` keyword to TCI2 construction and optimization:

```julia
crossinterpolate2(T, f, localdims; batchedf=nothing, kwargs...)
optimize!(tci, f; batchedf=nothing, kwargs...)
```

`f(indexset)` remains the scalar evaluator. `batchedf(indices)` is optional and, when supplied, is used for batch evaluation.

## Batch Layout

`batchedf` receives `indices::AbstractMatrix{<:Integer}` with shape:

```julia
(length(localdims), npoints)
```

Each column is one global multi-index point. The second dimension is the batch dimension. This matches the TreeTCI layout in `tensor4all-rs` and is memory-efficient in Julia because complete points are contiguous columns in column-major storage.

`batchedf` must return an `AbstractVector{T}` with length `npoints`.

## Dispatch Priority

Batch evaluation uses this priority:

1. If `batchedf !== nothing`, assemble the `(n_sites, npoints)` index matrix and call `batchedf`.
2. Otherwise, if `f isa BatchEvaluator`, use the existing inherited interface.
3. Otherwise, fall back to scalar `f(indexset)` loops.

`checkbatchevaluatable=true` should accept either `batchedf !== nothing` or `f isa BatchEvaluator`.

## Internal Ordering

For the current TCI2 tensor-fill path, points should be assembled in the same order as the existing result layout:

```julia
(left, center..., right)
```

The left index varies fastest, then center grid indices, then right index. This allows the returned values to be reshaped directly to:

```julia
(length(leftindexset), center_dims..., length(rightindexset))
```

## Compatibility

The existing `BatchEvaluator`, `ThreadedBatchEvaluator`, and `CachedFunction` behavior should keep working. Documentation should mark the function-based `batchedf` API as the recommended path and describe the inherited API as a compatibility interface.

## Testing

Use TDD. Add failing tests first for:

- `_batchevaluate_dispatch` with `batchedf`, verifying matrix shape, column ordering, and values.
- `crossinterpolate2(...; batchedf=...)` with a plain scalar `f` and no inherited type.
- `checkbatchevaluatable=true` accepting `batchedf`.
- Existing `BatchEvaluator` tests continuing to pass.

Cached batch evaluation can be updated after the core TCI2 path is stable, unless tests show it is required by the new API.
