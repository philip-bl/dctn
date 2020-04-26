import torch

from dctn.eps import eps
from dctn.epses_composition import inner_product


def test_inner_product() -> None:
    a = torch.einsum("oi,j->ijo", torch.eye(3), torch.ones(3))
    assert torch.allclose(inner_product((a,), (a,)), torch.tensor(9.0))
    assert torch.allclose(inner_product((a, a), (a, a)), torch.tensor(3.0 ** 4))
    assert torch.allclose(inner_product((a, a, a), (a, a, a)), torch.tensor(3.0 ** 8))

    green_mat = torch.eye(6)[:4]  # 4×6
    green_vec = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    black_mat = torch.eye(4)[:3]  # 3×4
    black_vec = torch.tensor([1.5, 0.0, 0.0, 0.0])

    orange_mat = green_mat
    orange_vec = torch.ones(6)

    red_mat = torch.eye(4)[1:]  # 3×4
    red_vec = torch.tensor([1.0, 0.0, 0.0, 1.0])

    green_eps = torch.einsum("oj,i->ijo", green_mat, green_vec)
    black_eps = torch.einsum("oi,j->ijo", black_mat, black_vec)
    orange_eps = torch.einsum("oi,j->ijo", orange_mat, orange_vec)
    red_eps = torch.einsum("oi,j->ijo", red_mat, red_vec)

    epses1 = (green_eps, black_eps)
    epses2 = (orange_eps, red_eps)

    # there are 3 disconnected subgraphs in the resulting tensor network
    first_multiplier = 2 + 3 + 4
    second_multiplier = 5
    third_multiplier = 1.5

    assert torch.allclose(
        inner_product(epses1, epses2),
        torch.tensor(first_multiplier * second_multiplier * third_multiplier),
    )
