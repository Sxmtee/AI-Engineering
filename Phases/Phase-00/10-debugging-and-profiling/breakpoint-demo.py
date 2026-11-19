import torch
from torch import nn

model = nn.Linear(3, 1)

for step in range(3):
    x = torch.randn(4, 3)
    loss = model(x).pow(2).mean()
    loss.backward()
    if step == 1:
        breakpoint()
    model.zero_grad()
    print(f"step {step} done, loss={loss.item():.4f}")
