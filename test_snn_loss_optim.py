
import torch
import matplotlib.pyplot as plt
from time import time
import numpy as np

import pytest

from soft_nearest_neighbors.optim import get_loss, grid_searches, find_working_lr
from soft_nearest_neighbors.loss import SoftNearestNeighbours


@pytest.mark.parametrize("cosine", [False, True])
def test_find_working_lr(cosine):
    print()
    torch.random.manual_seed(42)
    x, y = make_data(50, [0, 2], [-1, 1], 0.5)
    init_t, init_loss = grid_searches(x, y, [0.05, 1], [1, 50], [20, 10], cosine=cosine, use_gpu=True)
    losses, temps, flags = find_working_lr(x, y, 0.1, 200, init_t, cosine=cosine)
    assert flags["done"]


def test_find_working_lr_fails():
    print()
    torch.random.manual_seed(42)
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    losses, temps, flags = find_working_lr(x, y, 10000000.0, 200, 0.1)
    assert not flags["done"]


@pytest.mark.parametrize(
    "cosine, result", [(False, 0.6425162553787231), (True, 0.7328433990478516)]
)
def test_loss_returns_same_result(cosine, result):
    print()
    torch.random.manual_seed(42)
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    model = SoftNearestNeighbours(x, y, temperature_init=10, raise_on_inf=True, cosine=cosine)
    loss = model()
    assert loss.item() == result


@pytest.mark.parametrize("cosine, x1, y1", [(False, 0, 0), (True, -2, 2)])
def test_get_loss_runs(cosine, x1, y1):
    print()
    x, y = make_data(50, [x1, 2], [y1, 0], 0.5)
    t0 = time()
    losses, temps, flags = get_loss(x, y, 0.5, 200, 1.0, cosine=cosine)
    t1 = time()

    print(f"Time: {t1 - t0:.2f} s")

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(losses)
    ax2.plot(temps, color="C2")

    ax.set_ylabel("Loss")
    ax2.set_ylabel("Temperature")
    plt.tight_layout()
    plt.show()

    assert not np.isnan(losses).any()


@pytest.mark.parametrize("cosine", [False, True])
def test_optimised_loss_better_than_gridsearch(cosine):
    print()
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    init_t, init_loss = grid_searches(x, y, [0.05, 1], [1, 50], [20, 10], cosine=cosine)
    losses, temps, flags = get_loss(x, y, 0.5, 200, init_t=init_t, cosine=cosine)
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
