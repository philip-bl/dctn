from itertools import chain

import torch
import torch.nn.functional as F

import opt_einsum as oe

import einops

device = torch.device("cuda")

# dimensions are batch, channel, height, width, quantum
images = torch.rand(8, 3, 128, 128, 2, device=device)

# cores connectivity:
# o-o-o
#     |
# o-o-o
# |
# o-o-o

bond_dimension = 5
small_output_size = 1
large_output_size = 2

cores = {
    "upper_left": torch.rand(
        small_output_size, 1, bond_dimension, 2, 2, 2, device=device
    ),
    "upper_middle": torch.rand(
        small_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "upper_right": torch.randn(
        small_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "middle_right": torch.randn(
        large_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "middle_middle": torch.randn(
        large_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "middle_left": torch.randn(
        large_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "lower_left": torch.randn(
        small_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "lower_middle": torch.randn(
        small_output_size, bond_dimension, bond_dimension, 2, 2, 2, device=device
    ),
    "lower_right": torch.randn(
        small_output_size, bond_dimension, 1, 2, 2, 2, device=device
    ),
}

channels_separately = tuple(images[:, c] for c in range(3))
# core = cores["middle_middle"]
# foo = torch.einsum("bhwi,bhwj,bhwk,qxyijk->bhwqxy", *channels_separately, core)
cores_applied_to_subtensors = {
    # key: einops.rearrange(
    #     torch.einsum("bhwi,bhwj,bhwk,qxyijk->bqxyhw", *channels_separately, core),
    #     "b q x y h w -> b (q x y) h w",
    # )
    key: torch.einsum("bhwi,bhwj,bhwk,qxyijk->bqxyhw", *channels_separately, core)
    # b=batch, h=height, w=width, q=quantum mode of output, ijk=quantum modes of input, xy=bond modes
    for (key, core) in cores.items()
}

# I do batch snake tensor train contraction by doing padding, einsum, and then discarding the padding
def left_pad(key: str) -> int:
    if key.endswith("left"):
        return 2
    elif key.endswith("middle"):
        return 1
    else:
        return 0


def top_pad(key: str) -> int:
    if key.startswith("upper"):
        return 2
    elif key.startswith("middle"):
        return 1
    else:
        return 0


# paddings go as  (left, right, top, lower)
padding_sizes = {
    key: (left_pad(key), 2 - left_pad(key), top_pad(key), 2 - top_pad(key))
    for key in cores.keys()
}

padded = {
    key: F.pad(core, padding_sizes[key], value=float("nan"))
    for key, core in cores_applied_to_subtensors.items()
}
# now each padded has shape (batch, quantum output, 1st bond, 2nd bond, height, width


pre_result_dimensions = [
    "batch",
    *(f"quantumoutput{i}" for i in range(9)),
    "height",
    "width",
]
result = einops.rearrange(
    oe.contract(
        *chain.from_iterable(
            (
                padded[key],
                [
                    "batch",
                    f"quantumoutput{i}",
                    f"bond{i}",
                    f"bond{i+1}",
                    "height",
                    "width",
                ],
            )
            for i, key in enumerate(
                (
                    "upper_left",
                    "upper_middle",
                    "upper_right",
                    "middle_right",
                    "middle_middle",
                    "middle_left",
                    "lower_left",
                    "lower_middle",
                    "lower_right",
                )
            )
        ),
        pre_result_dimensions,
        optimize="optimal",
        backend="torch",
    ),
    f"{' '.join(pre_result_dimensions)} -> batch ( {' '.join(pre_result_dimensions[1:10])} ) height width",
)
# result has shape batch × quantum × h × w
assert torch.all(torch.isfinite(result[:, :, 2:-2, 2:-2]))
assert torch.any(torch.isnan(result))


einsum_expression = oe.contract_expression(
    *chain.from_iterable(
        (
            padded[key].shape,
            [
                "batch",
                f"quantumoutput{i}",
                f"bond{i}",
                f"bond{i+1}",
                "height",
                "width",
            ],
        )
        for i, key in enumerate(
            (
                "upper_left",
                "upper_middle",
                "upper_right",
                "middle_right",
                "middle_middle",
                "middle_left",
                "lower_left",
                "lower_middle",
                "lower_right",
            )
        )
    ),
    pre_result_dimensions,
    optimize="optimal",
)

foo = einops.rearrange(
einsum_expression(
    *(
        padded[key]
        for key in (
            "upper_left",
            "upper_middle",
            "upper_right",
            "middle_right",
            "middle_middle",
            "middle_left",
            "lower_left",
            "lower_middle",
            "lower_right",
        )
    ),
    backend="torch",
),
    f"{' '.join(pre_result_dimensions)} -> batch ( {' '.join(pre_result_dimensions[1:10])} ) height width"
)

assert torch.all(foo[:, :, 2:-2, 2:-2] == result[:, :, 2:-2, 2:-2])
