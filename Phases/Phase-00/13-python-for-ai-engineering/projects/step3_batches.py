import random
from itertools import islice


def batches(items, batch_size, shuffle=False, drop_last=False, seed=0):
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")

    if shuffle:
        items = list(items)
        random.Random(seed).shuffle(items)

    return _batch_iter(iter(items), batch_size, drop_last)


def _batch_iter(iterator, batch_size, drop_last):
    while True:
        batch = list(islice(iterator, batch_size))

        if not batch or (drop_last and len(batch) < batch_size):
            return

        yield batch


# Normal batching
print(list(batches(range(10), 4)))

# Drop incomplete batch
print(list(batches(range(10), 4, drop_last=True)))

# Generator support without shuffling
squares = (i * i for i in range(7))
print([len(b) for b in batches(squares, 3)])

# Reproducible shuffling
print(list(batches(range(10), 4, shuffle=True, seed=42)))
print(list(batches(range(10), 4, shuffle=True, seed=42)))

# Original data remains unchanged
items = list(range(10))
list(batches(items, 4, shuffle=True, seed=42))
print(items)
