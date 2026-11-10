# `batchedf` Batch Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a function-based batch evaluation API for TCI2 using `batchedf(indices::AbstractMatrix)` with the rightmost dimension as the batch dimension.

**Architecture:** Keep scalar `f(indexset)` as the primary mathematical function, and pass optional `batchedf` through TCI2 optimization and tensor filling. Add a new dispatch path in `src/batcheval.jl` that assembles `(n_sites, npoints)` index matrices and reshapes the returned vector into the existing TCI2 tensor layout, while preserving `BatchEvaluator` compatibility.

**Tech Stack:** Julia, TensorCrossInterpolation.jl, `Test`, existing TCI2 and batch evaluation tests.

---

### Task 1: Add Direct `batchedf` Dispatch Tests

**Files:**
- Modify: `test/test_batcheval.jl`
- Modify: `src/batcheval.jl`

**Step 1: Write the failing tests**

Add a new testset inside `@testset "batcheval"`:

```julia
@testset "batchedf matrix API" begin
    localdims = [2, 3, 2, 4]
    leftindexset = [[1], [2]]
    rightindexset = [[1], [3], [4]]
    seen_shape = Ref{Tuple{Int,Int}}()
    seen_indices = Ref{Matrix{Int}}()

    f = x -> sum(x)
    batchedf = indices -> begin
        seen_shape[] = size(indices)
        seen_indices[] = copy(indices)
        [sum(view(indices, :, p)) for p in axes(indices, 2)]
    end

    result = TCI._batchevaluate_dispatch(
        Float64,
        f,
        batchedf,
        localdims,
        leftindexset,
        rightindexset,
        Val(2),
    )

    ref = [
        sum(vcat(l, c, cp, r))
        for l in leftindexset,
            c in 1:localdims[2],
            cp in 1:localdims[3],
            r in rightindexset
    ]

    @test seen_shape[] == (length(localdims), length(result))
    @test result ≈ ref
    @test seen_indices[][:, 1] == [1, 1, 1, 1]
    @test seen_indices[][:, 2] == [2, 1, 1, 1]
end

@testset "batchedf validates output length" begin
    localdims = [2, 2, 2]
    f = x -> sum(x)
    bad_batchedf = indices -> ones(Float64, size(indices, 2) - 1)

    @test_throws DimensionMismatch TCI._batchevaluate_dispatch(
        Float64,
        f,
        bad_batchedf,
        localdims,
        [[1]],
        [[1]],
        Val(1),
    )
end
```

**Step 2: Run tests to verify they fail**

Run:

```bash
julia --project=. test/test_batcheval.jl
```

Expected: FAIL because `_batchevaluate_dispatch(::Type, f, batchedf, ...)` is not defined.

**Step 3: Implement minimal dispatch**

In `src/batcheval.jl`, add:

```julia
function _batchevaluate_dispatch(
    ::Type{V},
    f,
    batchedf,
    localdims::Vector{Int},
    leftindexset::AbstractVector{MultiIndex},
    rightindexset::AbstractVector{MultiIndex},
    ::Val{M},
)::Array{V,M + 2} where {V,M}
    if length(leftindexset) * length(rightindexset) == 0
        return Array{V,M + 2}(undef, ntuple(i -> 0, M + 2)...)
    end

    nl = length(first(leftindexset))
    nr = length(first(rightindexset))
    L = M + nl + nr
    center_dims = localdims[nl+1:L-nr]
    npoints = length(leftindexset) * prod(center_dims) * length(rightindexset)
    indices = Matrix{Int}(undef, L, npoints)

    p = 1
    for rightindex in rightindexset
        for cindex in Iterators.product(ntuple(x -> 1:localdims[nl+x], M)...)
            for leftindex in leftindexset
                indices[1:nl, p] .= leftindex
                indices[nl+1:nl+M, p] .= cindex
                indices[nl+M+1:end, p] .= rightindex
                p += 1
            end
        end
    end

    values = batchedf(indices)
    length(values) == npoints || throw(DimensionMismatch("batchedf returned $(length(values)) values for $npoints points"))
    return reshape(Vector{V}(values), length(leftindexset), center_dims..., length(rightindexset))
end
```

Keep the existing fallback methods unchanged.

**Step 4: Run tests to verify they pass**

Run:

```bash
julia --project=. test/test_batcheval.jl
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/batcheval.jl test/test_batcheval.jl
git commit -m "feat: add batchedf matrix dispatch"
```

### Task 2: Thread `batchedf` Through TCI2

**Files:**
- Modify: `test/test_tensorci2.jl`
- Modify: `src/tensorci2.jl`

**Step 1: Write the failing tests**

In `test/test_tensorci2.jl`, add tests near `"checkbatchevaluatable"`:

```julia
@testset "batchedf keyword" begin
    localdims = fill(2, 5)
    f(x) = Float64(sum(x))
    calls = Ref(0)
    batchedf = indices -> begin
        calls[] += 1
        [Float64(sum(view(indices, :, p))) for p in axes(indices, 2)]
    end

    tci, ranks, errors = crossinterpolate2(
        Float64,
        f,
        localdims;
        batchedf,
        tolerance=1e-12,
        maxiter=2,
        checkbatchevaluatable=true,
    )

    @test calls[] > 0
    @test evaluate(tci, ones(Int, length(localdims))) ≈ f(ones(Int, length(localdims)))
end
```

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project=. test/test_tensorci2.jl
```

Expected: FAIL because `batchedf` is an unexpected keyword or `checkbatchevaluatable` still rejects plain `f`.

**Step 3: Implement minimal TCI2 plumbing**

In `src/tensorci2.jl`:

- Add `batchedf=nothing` keyword to `optimize!`.
- Change `checkbatchevaluatable` validation to:

```julia
if checkbatchevaluatable && isnothing(batchedf) && !(f isa BatchEvaluator)
    error("Function `f` is not batch evaluatable")
end
```

- Use a local evaluator:

```julia
batchf = isnothing(batchedf) ? f : BatchEvaluatorAdapter{ValueType}(f, batchedf, collect(sitedims(tci)))
```

or add a new wrapper name if clearer.

- Pass `batchf` to `sweep2site!`, `fillsitetensors!`, global pivot search, and error checks where the existing code expects batch-capable evaluation but still needs scalar calls.
- Add `batchedf=nothing` keyword to `crossinterpolate2` and forward it to `optimize!(tci, f; batchedf, kwargs...)`.

Prefer a wrapper that implements both scalar `f(indexset)` and batch dispatch through `batchedf`.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --project=. test/test_tensorci2.jl
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/tensorci2.jl test/test_tensorci2.jl
git commit -m "feat: support batchedf in tci2"
```

### Task 3: Clean Up Adapter Naming And Compatibility

**Files:**
- Modify: `src/batcheval.jl`
- Modify: `test/test_batcheval.jl`
- Modify: `test/test_cachedfunction.jl` if required by failures

**Step 1: Write or adjust compatibility tests**

Ensure existing tests still cover:

```julia
bf = TCI.makebatchevaluatable(Float64, tbf, localdims)
@test size(bf(leftindexset, rightindexset, Val(1))) == (2, 3, 2)
```

Add a direct wrapper test if a new wrapper is introduced:

```julia
wrapped = TCI.makebatchevaluatable(Float64, f, batchedf, localdims)
@test TCI.isbatchevaluable(wrapped)
@test wrapped([1, 1, 1]) == f([1, 1, 1])
@test wrapped([[1]], [[1]], Val(1)) ≈ reshape(batchedf([1 1; 1 2; 1 1]), 1, 2, 1)
```

**Step 2: Run focused tests**

Run:

```bash
julia --project=. test/test_batcheval.jl test/test_cachedfunction.jl
```

Expected: FAIL if wrapper constructors or `CachedFunction` assumptions are inconsistent.

**Step 3: Refactor implementation**

In `src/batcheval.jl`, keep `makebatchevaluatable(::Type{T}, f, localdims)` unchanged and add:

```julia
makebatchevaluatable(::Type{T}, f, batchedf, localdims) where {T} = BatchEvaluatorAdapter{T}(f, batchedf, localdims)
```

Update `BatchEvaluatorAdapter` fields to support optional `batchedf`, or add a separate wrapper if that keeps constructors clearer.

**Step 4: Run focused tests**

Run:

```bash
julia --project=. test/test_batcheval.jl test/test_cachedfunction.jl
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/batcheval.jl test/test_batcheval.jl test/test_cachedfunction.jl
git commit -m "refactor: keep batch evaluator compatibility"
```

### Task 4: Update Documentation

**Files:**
- Modify: `docs/src/index.md`

**Step 1: Update docs**

In the batch evaluation section, replace the inheritance-first example with:

```julia
f(indexset) = sum(indexset)

batchedf = indices -> [
    sum(view(indices, :, p))
    for p in axes(indices, 2)
]

tci, ranks, errors = TCI.crossinterpolate2(
    Float64,
    f,
    localdims;
    batchedf,
)
```

Document that `indices` has shape `(length(localdims), npoints)` and each column is one point. Keep a short compatibility note for `BatchEvaluator`.

**Step 2: Run docs-related check if available**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS. If too slow, at least run all affected tests from prior tasks.

**Step 3: Commit**

```bash
git add docs/src/index.md
git commit -m "docs: document batchedf batch evaluation"
```

### Task 5: Full Verification

**Files:**
- No source edits expected

**Step 1: Run full test suite**

Run:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: PASS.

**Step 2: Inspect final diff**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: clean worktree and recent commits for design, implementation, compatibility, and docs.
