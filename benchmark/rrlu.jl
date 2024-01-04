# Written by Satoshi Terasaki

import TensorCrossInterpolation as TCI
using Plots
using LinearAlgebra
BLAS.set_num_threads(1)

Ns = [100, 500, 1000, 2000]

timings = zeros(Float64, length(Ns), 2)

for (i, N) in enumerate(Ns)
    A = rand(N, N)
    # warm up
    lu_ = TCI.rrlu(A)

    td_rrlu = @timed TCI.rrlu(A)
    td_lu = @timed lu(A)
    @info "N=$(N)" td_rrlu.time td_lu.time
    timings[i, 1] = td_rrlu.time
    timings[i, 2] = td_lu.time
end

p = plot(
    xaxis=:log,
    yaxis=:log,
    xticks=Ns,
    yticks=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
    xlabel="N",
    ylabel="time (s)",
    legend=:bottomright
)
plot!(p, Ns, timings[:, 1], label="rrlu", marker=:o)
plot!(p, Ns, timings[:, 2], label="lu", marker=:x)
plot!(p, Ns, 1e-6 .* Ns .^ 3, label="cubic")

savefig("bench_rrlu.png")
