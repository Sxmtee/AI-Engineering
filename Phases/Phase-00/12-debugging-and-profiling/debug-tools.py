import time

import pandas as pd
import torch
from torch import nn


def debug_print(name, tensor):
    print(
        f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
        f"mean={tensor.mean().item():.4f}, "
        f"has_nan={tensor.isnan().any().item()}"
    )


def check_shapes(model, sample_input):
    print(f"Input: {sample_input.shape}")
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            in_shape = inp[0].shape if isinstance(inp, tuple) else inp.shape
            out_shape = out.shape if hasattr(out, "shape") else type(out)
            print(f"  {name}: {in_shape} -> {out_shape}")

        return hook

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()


def detect_nan(model, loss, step):
    if torch.isnan(loss):
        print(f"NaN loss at step {step}")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  NaN gradient in {name}")
                if torch.isinf(param.grad).any():
                    print(f"  Inf gradient in {name}")
        return True
    return False


class Timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")


def check_data_leakage(train_set, test_set, id_column="id"):
    train_ids = set(train_set[id_column].tolist())
    test_ids = set(test_set[id_column].tolist())
    overlap = train_ids & test_ids
    if overlap:
        print(f"DATA LEAKAGE: {len(overlap)} samples in both train and test")
        return True
    return False


if __name__ == "__main__":
    x = torch.randn(4, 3)
    debug_print("good tensor", x)
    bad = torch.tensor([1.0, float("nan"), 3.0])
    debug_print("bad tensor", bad)
    print("\n--- shape check ---")
    model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
    check_shapes(model, torch.randn(4, 3))
    print("\n--- nan detection ---")
    model = nn.Linear(3, 1)
    out = model(torch.randn(4, 3)) / 0  # divide by zero on purpose
    loss = (out - out).mean()  # inf - inf = nan
    loss.backward()
    detect_nan(model, loss, step=1)
    print("\n--- timing ---")
    model = nn.Sequential(nn.Linear(1000, 1000), nn.ReLU(), nn.Linear(1000, 10))
    with Timer("data creation"):
        batch = torch.randn(512, 1000)
    with Timer("forward pass"):
        out = model(batch)
    with Timer("backward pass"):
        out.sum().backward()
    print("\n--- data leakage ---")
    train = pd.DataFrame({"id": [1, 2, 3, 4, 5], "x": range(5)})
    leaky_test = pd.DataFrame({"id": [4, 5, 6, 7], "x": range(4)})
    clean_test = pd.DataFrame({"id": [6, 7, 8], "x": range(3)})
    print("leak found:", check_data_leakage(train, leaky_test))
    print("leak found:", check_data_leakage(train, clean_test))
