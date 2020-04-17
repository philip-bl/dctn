import torch
import opt_einsum as oe

device = "cuda"
big_core = torch.randn(
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, dtype=torch.float64, device=device
)
batches_of_small_cores = [
    torch.randn(512, 25, 25, 2, dtype=torch.float64, device=device) for _ in range(16)
]

equation = "αβγi,αβγj,αβγk,αβγl,αβγm,αβγn,αβγo,αβγp,αβγq,αβγr,αβγs,αβγt,αβγu,αβγv,αβγw,αβγx,ijklmnopqrstuvwxω->αβγω"
print(oe.contract_path(equation, *batches_of_small_cores, big_core, optimize="auto"))
result = oe.contract(equation, *batches_of_small_cores, big_core, optimize="auto")
