import random


def train_val_test_split(items, train_fraction=0.7, seed=0):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if len(items) < 3:
        raise ValueError("need at least 3 items")

    # 1. Create and shuffle the indices
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)

    # 2. Calculate the number of training items
    n_train = min(max(1, round(len(items) * train_fraction)), len(items) - 2)

    # 3. Calculate how many items remain
    remaining = len(items) - n_train

    # 4. Divide the remaining items between validation and test
    n_val = remaining // 2
    # n_test = remaining - n_val

    # 5. Slice the shuffled indices into three groups
    train_indices = [items[i] for i in order[:n_train]]
    val_indices = [items[i] for i in order[n_train : n_train + n_val]]
    test_indices = [items[i] for i in order[n_train + n_val :]]

    # 6. Convert indices back into items
    train = train_indices
    validation = val_indices
    test = test_indices

    return train, validation, test


items = list(range(20))

train, validation, test = train_val_test_split(items, seed=42)

print("Training:", train, len(train))
print("Validation:", validation, len(validation))
print("Test:", test, len(test))
