
import torch
import matplotlib.pyplot as plt
from time import time

from soft_nearest_neighbors.loss import SoftNearestNeighbours


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


def test_loss_runs():
    x, y = make_data(50, [0, 2], [0, 0], 0.5)
    t0 = time()
    losses, temps = get_loss(x, y, 0.1, 100)
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

    print("Done!")


def training_loop(model, optimizer, n=20):
    losses = []
    temps = [model.weights.detach().cpu().numpy()[0]]
    for i in range(n):
        loss = model()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        temps.append(model.weights.detach().cpu().numpy()[0])
        del loss

    return losses, temps


def get_loss(x, y, lr=0.1, n=20, init_t=0.1):
    model = SoftNearestNeighbours(x.to(device), y.to(device), temperature_init=init_t)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return training_loop(model, optimizer, n)


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
