
import torch
import matplotlib.pyplot as plt
from time import time
import numpy as np

from soft_nearest_neighbors.optim import get_loss, grid_searches
from soft_nearest_neighbors.loss import SoftNearestNeighbours


def test_loss_returns_same_result():
    print()
    torch.random.manual_seed(42)
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    model = SoftNearestNeighbours(x, y, temperature_init=10, raise_on_inf=True)
    loss = model()
    assert loss.item() == 0.6425162553787231


def test_get_loss_runs():
    print()
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    t0 = time()
    losses, temps, flags = get_loss(x, y, 0.5, 100, 1.0)
    t1 = time()

    print(f"Time: {t1 - t0:.2f} s")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(losses)
    ax2.plot(temps, color="C2")

    ax.set_ylabel("Loss")
    ax2.set_ylabel("1/Temperature")
    plt.tight_layout()
    plt.show()

    assert not np.isnan(losses).any()


def test_optimised_loss_better_than_gridsearch():
    print()
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    init_loss, init_t = grid_searches(x, y, [0.05, 1], [1, 50], [20, 10])
    losses, temps, flags = get_loss(x, y, 0.5, 100, init_t=init_t)
    assert losses[-1] < init_loss


def make_data(n, x_means, y_means, std):
    n_classes = len(x_means)
    means = torch.zeros((n_classes, n, 2))
    for i in range(n_classes):
        means[i, :, 0] = x_means[i]
        means[i, :, 1] = y_means[i]
    means = means.reshape((n * n_classes, 2))
    labels = torch.cat([torch.zeros(n, dtype=int) + i for i in range(n_classes)])
    data = torch.normal(means, std)
    return data, labels
