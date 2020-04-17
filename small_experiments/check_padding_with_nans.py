import torch

x = torch.tensor([1.0, 1.0], requires_grad=True)
y = torch.tensor([2.0, float("nan")], requires_grad=True)

result = (x * y)[0]
assert torch.isfinite(result)
result.backward()
assert not torch.all(torch.isfinite(x.grad))
print(x.grad)
