# Implementation of the TCI algorithm

This document gives an overview to this implementation of the TCI algorithm.
First, we will show how the high-level components work together to produce a
TCI without going into detail; a detailed introduction of each component
follows.

## Overall structure

```@docs
TensorCrossInterpolation.crossinterpolate
```

This function takes a target function to be approximated and constructs a TCI up to some specified tolerance using a sweeping algorithm. The steps are as follows:

- Initialize the TCI structure (`TensorCI`) using `f(firstpivot)`.
- Iterate for `iter = 1...maxiter`
    - If `iter` is odd, sweep forward, else, sweep backward. During the sweep, add one pivot to each link using `addpivot!`.
    - Update `errornormalization` to the maximum sample so far.
    - If `max(pivoterrors) / errornormalization < tolerance`, abort iteration.
- Return the TCI.

## Initialization

TODO.

## Data to keep track of / Glossary of member variables

TODO.

## Sweeps

Sweeps are done by applying `addpivot!` to each link $\ell = 1\ldots\mathscr{L}$ in ascending order for forward sweeps and descending order for backward sweeps.

```@docs
TensorCrossInterpolation.addpivot!
```
This function adds one pivot at bond $\ell$ (in the code, we use `p` instead of $\ell$). This is done as follows:

- If $\operatorname{rank}(ACA_\ell) \geq \min(nrows(\Pi_\ell), ncols(\Pi_\ell))$: Skip this bond and proceed with the next one, since we're already at full rank.
- $E \leftarrow \lvert ACA_\ell - \Pi_\ell \rvert$
- $(i, j) \leftarrow \argmax E$
- `pivoterrors`$_\ell \leftarrow E[i, j]$
- If $E[i, j] < \tau_{\text{pivot}} (= 10^{-16})$, skip this $\ell$.
- Otherwise, add $(i, j)$ as a new pivot to this bond $\ell$ (see below).

## Adding a pivot $(i, j)$ to bond $\ell$


```@docs
TensorCrossInterpolation.addpivotcol!
TensorCrossInterpolation.addpivotrow!
```

To add a pivot, we have to add a row and a column to $T_\ell, P_\ell$ and $T_{\ell + 1}$. Afterwards, we update neighbouring $\Pi$ tensors $\Pi_{\ell-1}$ and $\Pi_{\ell+1}$ for efficiency.

- Construct an $MCI$ (`MatrixCI`) object with row indices $I$, column indices $J$, columns $C$ and rows $R$, where:
    - $MCI.I \leftarrow \Pi I_\ell [I_{\ell+1}]$
    - $MCI.J \leftarrow \Pi J_{\ell+1} [J_\ell]$
    - $MCI.C \leftarrow \text{reshape}(T_\ell; D_{\ell-1}\times d_\ell, D_{\ell})$
    - $MCI.R \leftarrow \text{reshape}(T_{\ell+1}; D_{\ell}, d_{\ell+1} \times D_{\ell+1})$
- Add the column $j$ to the bond, like this:
    - add $j$ to $ACA_\ell$
    - add $j$ to $MCI$
    - push $\Pi J_{\ell+1}[j]$ to $J_\ell$
    - $T_\ell \leftarrow \text{reshape}(MCI.C;D_{\ell-1}, d_\ell, D_{\ell})$
    (this will split the legs again)
    - $P_\ell \leftarrow MCI.P$ ($P$ is obtained implicitly as a submatrix of $MCI.C$.)
    - update columns of $\Pi_{\ell-1}$
- Add the row $i$ to the bond, like this:
    - add $i$ to $ACA_\ell$
    - add $i$ to $MCI$
    - push $\Pi I_\ell[i]$ to $I_{\ell+1}$
    - $T_{\ell+1} \leftarrow \text{reshape}(MCI.R; D_\ell, d_{\ell+1}, D_{\ell+1})$
    - $P_\ell \leftarrow MCI.P$
    - update rows of $\Pi_{\ell+1}$.

