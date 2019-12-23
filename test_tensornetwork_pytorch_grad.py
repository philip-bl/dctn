import torch
import tensornetwork as tn

tn.set_default_backend("pytorch")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a_tensor = (torch.eye(3, device=device) * -2.0).requires_grad_(True)
b_tensor = (torch.eye(3) + torch.randn(3, 3)).to(device).requires_grad_(True)
c_tensor = (torch.eye(3, device=device) * 3.0).requires_grad_(True)

parameters = (a_tensor, b_tensor, c_tensor)


lr = 0.0003

for i in range(100):
    nodes = set()
    with tn.NodeCollection(nodes):
        a = tn.Node(a_tensor, "a")
        b = tn.Node(b_tensor, "b")
        c = tn.Node(c_tensor, "c")
        a[1] ^ b[0]
        b[1] ^ c[0]
        c[1] ^ a[0]
    
    result = tn.contractors.auto(nodes).tensor
    target = 0.3
    loss = ((result - target) ** 2)
    print(loss.item())

    for tensor in parameters:
        tensor.grad = None
    loss.backward()
    for tensor in parameters:
        tensor.data -= lr * tensor.grad
