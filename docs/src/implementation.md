```@meta
CurrentModule = TensorCrossInterpolation
```

# Implementation

This document gives an overview to this implementation of the TCI algorithm.
First, we will show how the high-level components work together to produce a
TCI without going into detail; a detailed introduction of each component
follows.

## Overall structure

### [`crossinterpolate`](@ref)

This function takes a target function to be approximated and constructs a TCI up to some specified tolerance using a sweeping algorithm. The steps are as follows:

- Initialize the TCI structure ([`TensorCI`](@ref)) using `f(firstpivot)`.
- Iterate for `iter = 1...maxiter`
    - If `iter` is odd, sweep forward, else, sweep backward. During the sweep, add one pivot to each link using [`addpivot!`](@ref).
    - Update `errornormalization` to the maximum sample so far.
    - If `max(pivoterrors) / errornormalization < tolerance`, abort iteration.
- Return the TCI.

## Initialization

TODO.

## Data to keep track of / Glossary of member variables

TODO.

## Sweeps

Sweeps are done by applying [`addpivot!`](@ref) to each link $\ell = 1\ldots\mathscr{L}$ in ascending order for forward sweeps and descending order for backward sweeps.

### [`addpivot!`](@ref)
This function adds one pivot at bond $\ell$ (in the code, we use `p` instead of $\ell$). This is done as follows:

- If $\operatorname{rank}(ACA_\ell) \geq \min(nrows(\Pi_\ell), ncols(\Pi_\ell))$: Skip this bond and proceed with the next one, since we're already at full rank.
- Evaluate the error matrix $E \leftarrow \lvert ACA_\ell - \Pi_\ell \rvert$.
- New pivot is at the maximum error $(i, j) \leftarrow \argmax E$
- Save the last pivot error to `pivoterrors`$_\ell \leftarrow E[i, j]$
- If $E[i, j] < \tau_{\text{pivot}} (= 10^{-16})$, skip this $\ell$.
- Otherwise, add $(i, j)$ as a new pivot to this bond $\ell$ (see below).

## Adding a pivot $(i, j)$ to bond $\ell$

### [`addpivotcol!`](@ref) and [`addpivotrow!`](@ref)

To add a pivot, we have to add a row and a column to $T_\ell, P_\ell$ and $T_{\ell + 1}$. Afterwards, we update neighbouring $\Pi$ tensors $\Pi_{\ell-1}$ and $\Pi_{\ell+1}$ for efficiency.

- Construct an $MCI$ ([`MatrixCI`](@ref)) object with row indices $I$, column indices $J$, columns $C$ and rows $R$, where:
    - Row indices $MCI.I \leftarrow \Pi I_\ell [I_{\ell+1}]$
    - Column indices $MCI.J \leftarrow \Pi J_{\ell+1} [J_\ell]$
    - Column vectors $MCI.C \leftarrow \text{reshape}(T_\ell; D_{\ell-1}\times d_\ell, D_{\ell})$
    - Row vectors $MCI.R \leftarrow \text{reshape}(T_{\ell+1}; D_{\ell}, d_{\ell+1} \times D_{\ell+1})$
- Add the column $j$ to the bond, like this:
    - add $j$ to $ACA_\ell$
    - add $j$ to $MCI$
    - push $\Pi J_{\ell+1}[j]$ to $J_\ell$
    - Split the legs of $T_\ell \leftarrow \text{reshape}(MCI.C;D_{\ell-1}, d_\ell, D_{\ell})$
    - Update $P_\ell \leftarrow MCI.P$, where $MCI.P$ is obtained implicitly as a submatrix of $MCI.C$.
    - Update columns of $\Pi_{\ell-1}$
- Add the row $i$ to the bond, like this:
    - add $i$ to $ACA_\ell$
    - add $i$ to $MCI$
    - push $\Pi I_\ell[i]$ to $I_{\ell+1}$
    - Update $T_{\ell+1} \leftarrow \text{reshape}(MCI.R; D_\ell, d_{\ell+1}, D_{\ell+1})$
    - Update $P_\ell \leftarrow MCI.P$
    - Update rows of $\Pi_{\ell+1}$.

