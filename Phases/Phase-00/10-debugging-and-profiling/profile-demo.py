import time

import torch
from torch import nn


def load_batch():
    time.sleep(0.02)  # pretend this is slow disk or preprocessing
    return torch.randn(256, 100), torch.randn(256, 1)


def train_step(model, optimizer, x, y):
    loss = (model(x) - y).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    model = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(50):
        x, y = load_batch()
        train_step(model, optimizer, x, y)


if __name__ == "__main__":
    main()
