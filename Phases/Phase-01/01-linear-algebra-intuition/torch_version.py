import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x}")
print(f"y = {y}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")


p = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
n = torch.tensor([1.0, 2.0, 3.0])

similarity = torch.dot(p, n)
similarity.backward()

print("Dot product:", similarity.item())
print("Gradient:", p.grad)


w = torch.tensor(10.0, requires_grad=True)

target = 3.0
learning_rate = 0.1

loss = (w - target) ** 2

loss.backward()

print("Weight:", w.item())
print("Loss:", loss.item())
print("Gradient:", w.grad)
