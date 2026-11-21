import tracemalloc


def load_data():
    small = [i for i in range(1000)]
    big = [float(i) for i in range(1_000_000)]
    medium = {i: str(i) for i in range(50_000)}
    return small, big, medium


tracemalloc.start()
data = load_data()
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics("lineno")[:5]:
    print(stat)
