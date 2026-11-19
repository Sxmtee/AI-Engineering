import torch
import torch.utils.tensorboard.writer
from torch import nn

torch.manual_seed(0)
w = torch.randn(10, 1)
x_train, x_val = torch.randn(30, 10), torch.randn(30, 10)
y_train = x_train @ w + 0.5 * torch.randn(30, 1)
y_val = x_val @ w + 0.5 * torch.randn(30, 1)

model = nn.Sequential(
    nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
writer = torch.utils.tensorboard.writer.SummaryWriter("runs/overfit_demo")

for epoch in range(300):
    loss = loss_fn(model(x_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val)
    writer.add_scalar("loss/train", loss.item(), epoch)
    writer.add_scalar("loss/val", val_loss.item(), epoch)

writer.close()
print("done, logs in runs/overfit_demo")
