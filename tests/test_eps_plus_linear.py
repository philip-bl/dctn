from itertools import product

import torch

from dctn.eps_plus_linear import (
    ManuallyChosenInitialization,
    ZeroCenteredNormalInitialization,
    ZeroCenteredUniformInitialization,
    EPSesPlusLinear,
)


def test_epses_plus_linear_manually_chosen_initialization():
    epses_specs = ((4, 4), (3, 4), (3, 6))
    initialization = ManuallyChosenInitialization(
        (
            ZeroCenteredNormalInitialization(0.1),
            ZeroCenteredUniformInitialization(77.0),
            ZeroCenteredNormalInitialization(10.0),
        ),
        ZeroCenteredUniformInitialization(500.0),
        ZeroCenteredNormalInitialization(1e-6),
    )
    ps = (1e-3, 0.4, 0.6, 1.0)
    device = torch.device("cpu")
    dtypes = (torch.float32, torch.float64)

    for p, dtype in product(ps, dtypes):
        model = EPSesPlusLinear(epses_specs, initialization, p, device, dtype)
        assert 0.09 <= model.epses[0].std() <= 0.11
        assert -77.0 <= model.epses[1].min() <= -70.0
        assert 70.0 <= model.epses[1].max() <= 77
        assert 9.0 <= model.epses[2].std() <= 11.0
        assert -500.0 <= model.linear.weight.min() <= -460.0
        assert 460.0 <= model.linear.weight.max() <= 500.0
        assert 1e-9 <= model.linear.bias.std() <= 1e-3
