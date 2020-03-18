from itertools import chain

from more_itertools import windowed

import torch

import opt_einsum as oe


tt_rank = (1, 10)
dangling_dim_sizes = (2, 100)
cores = tuple(
    (torch.randn(dangl, lb, rb) * 3.0) / dangl
    #torch.ones(dangl, lb, rb) / dangl
    for (dangl, (lb, rb)) in zip(
        dangling_dim_sizes, chain(windowed(tt_rank, 2), ((tt_rank[-1], tt_rank[0]),))
    )
)
for core in cores:
    core.requires_grad = True
explicit = oe.contract(
    *chain.from_iterable(
        (
            (core, (f"i{c}", f"b{c}", f"b{c+1 if c < len(cores) else 0}"))
            for c, core in enumerate(cores)
        )
    ),
    tuple(f"i{c}" for c in range(len(cores))),
    optimize="auto",
)
explicit.sum().backward()
for c, core in enumerate(cores):
    print(f"{c}: {torch.mean(torch.abs(core.grad))}")
