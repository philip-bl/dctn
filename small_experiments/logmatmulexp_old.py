from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def logmatmulexp_slightly_stable_auto_backward(A: Tensor, B: Tensor) -> Tensor:
    """Calculates the following function, but in a numerically stable way:
  (A.exp() @ B.exp()).log()"""
    # A has shape ϴ×R, B has shape R×I
    A_max = A.max(dim=1, keepdim=True)[0]  # shape: (ϴ, 1)
    B_max = B.max(dim=0, keepdim=True)[0]  # shape: (1, I)
    blah = torch.log((A - A_max).exp() @ (B - B_max).exp())  # shape: (ϴ, I)
    return blah + A_max + B_max


def logmatmulexp_naive(A: Tensor, B: Tensor) -> Tensor:
    return (A.exp() @ B.exp()).log()


class LogMatMulExpSlightlyStableManualBackward(torch.autograd.Function):
    """Given matrix A of shape ϴ×R and matrix b of shape R×I, calculates
  (A.exp() @ B.exp()).log() and its backward in an efficient and numerically stable way."""

    @staticmethod
    def forward(ctx, A: Tensor, B: Tensor) -> Tensor:
        assert A.ndim == 2 and B.ndim == 2 and A.shape[1] == B.shape[0]
        A_max = A.max(dim=1, keepdim=True)[0]  # shape: (ϴ, 1)
        B_max = B.max(dim=0, keepdim=True)[0]  # shape: (1, I)

        A_stable_exp = (A - A_max).exp()
        # ∀θ ∀r A_stable_exp[θ,r] == (A[θ,r]-A[θ].max()).exp()

        B_stable_exp = (B - B_max).exp()
        # ∀i ∀r B_stable_exp[r,i] == (B[r,i]-B[:,i].max()).exp()

        AB_stable_exp = A_stable_exp @ B_stable_exp  # shape: (ϴ, I)
        # ∀θ ∀i AB_stable_exp[θ,i] = ∑_r exp(A[θ,r]-A[θ].max())*exp(B[r,i]-B[:,i].max())

        ctx.save_for_backward(A_stable_exp, B_stable_exp, AB_stable_exp)
        return AB_stable_exp.log() + A_max + B_max  # shape: (ϴ, I)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        A_stable_exp, B_stable_exp, AB_stable_exp = ctx.saved_tensors
        foo = grad_output / AB_stable_exp  # shape: (ϴ, I)
        return (
            A_stable_exp * (foo @ B_stable_exp.T) if ctx.needs_input_grad[0] else None,
            B_stable_exp * (A_stable_exp.T @ foo) if ctx.needs_input_grad[1] else None,
        )


def logmatmulexp_stable(log_A: Tensor, log_B: Tensor) -> Tensor:
    ϴ, R = log_A.shape
    assert R == log_B.shape[0]
    I = log_B.shape[1]
    log_A_expanded = log_A.unsqueeze(2).expand((ϴ, R, I))
    log_B_expanded = log_B.unsqueeze(0).expand((ϴ, R, I))
    log_pairwise_products = log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)
    max_log_pairwise_products = log_pairwise_products.max(dim=1, keepdim=True)[
        0
    ]  # shape: (ϴ, 1, I)
    stabilized_matmulexp = (
        (log_pairwise_products - max_log_pairwise_products).exp().sum(dim=1)
    )  # shape: (ϴ, I)
    return stabilized_matmulexp.log() + max_log_pairwise_products.squeeze(1)


class LogMatMulExpStableMemorySaving(torch.autograd.Function):
    """Given matrix log_A of shape ϴ×R and matrix log_B of shape R×I, calculates
  (log_A.exp() @ log_B.exp()).log() and its backward in a way which is numerically
  stable and doesn't save a tensor of size ϴ×R×I - this is why it's called
  MemorySaving. The backward method is implemented manually purely for the purpose
  of saving memory."""

    @staticmethod
    def _calc_log_pairwise_products(log_A: Tensor, log_B: Tensor) -> Tensor:
        ϴ, R = log_A.shape
        I = log_B.shape[1]
        assert log_B.shape == (R, I)
        log_A_expanded = log_A.unsqueeze(2).expand((ϴ, R, I))
        log_B_expanded = log_B.unsqueeze(0).expand((ϴ, R, I))
        return log_A_expanded + log_B_expanded  # shape: (ϴ, R, I)

    @staticmethod
    def forward(ctx, log_A: Tensor, log_B: Tensor) -> Tensor:
        ctx.save_for_backward(log_A, log_B)
        log_pairwise_products = LogMatMulExpStableMemorySaving._calc_log_pairwise_products(
            log_A, log_B
        )
        return torch.logsumexp(log_pairwise_products, dim=1)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        log_A, log_B = ctx.saved_tensors
        log_pairwise_products = LogMatMulExpSlightlyStableManualBackward._calc_log_pairwise_products(
            log_A, log_B
        )
        jacobian = F.softmax(log_pairwise_products, dim=1)  # shape: (ϴ, R, I)
        return (
            # torch.einsum("θri,θi->θr" if ctx.needs_input_grad[0] else None
            # TODO: no, don't do this by torch.einsum - it's fucking slow
        )


def compare_tensors(title: str, tensor1: Tensor, tensor2: Tensor) -> None:
    return_instantly = False
    if not torch.all(torch.isfinite(tensor1)):
        print(f"{title} has NaN or inf in tensor1")
        return_instantly = True
    if not torch.all(torch.isfinite(tensor2)):
        print(f"{title} has NaN or inf in tensor2")
        return_instantly = True
    if return_instantly:
        return
    if not torch.allclose(tensor1, tensor2):
        abs_err = (tensor1 - tensor2).abs()
        max_abs_err = abs_err.max()
        mean_abs_err = abs_err.mean()
        rel_err = abs_err / tensor2.abs()
        max_rel_err = rel_err.max()
        mean_rel_err = rel_err.mean()
        print(
            f"{title} differs. "
            f"{max_rel_err=:.2e}, {mean_rel_err=:.2e}, "
            f"{max_abs_err=:.2e}, {mean_abs_err=:.2e}."
        )
    else:
        print(f"{title} allclose.")


def compare_logmatmulexp(func1, func2, A: Tensor, B: Tensor, out_grad: Tensor) -> None:
    A1 = A.clone().detach().requires_grad_()
    B1 = B.clone().detach().requires_grad_()
    AB1 = func1(A1, B1)
    AB1.backward(out_grad)

    A2 = A.clone().detach().requires_grad_()
    B2 = B.clone().detach().requires_grad_()
    AB2 = func2(A2, B2)
    AB2.backward(out_grad)

    compare_tensors("AB", AB1, AB2)
    compare_tensors("A.grad", A1.grad, A2.grad)
    compare_tensors("B.grad", B1.grad, B2.grad)


c = 150
A = (torch.randn(100, 100, dtype=torch.float64) * c).requires_grad_()
B = (torch.randn(100, 100, dtype=torch.float64) * c).requires_grad_()
out_grad = torch.randn(100, 100, dtype=torch.float64) * c
compare_logmatmulexp(logmatmulexp_stable, logmatmulexp_naive, A, B, out_grad)
