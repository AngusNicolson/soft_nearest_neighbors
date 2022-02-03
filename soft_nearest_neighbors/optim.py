import numpy as np
import torch

from soft_nearest_neighbors.loss import SoftNearestNeighbours


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


def get_loss(x, y, lr=0.1, n=20, init_t=0.1, use_gpu=False):
    device = get_device(use_gpu)
    model = SoftNearestNeighbours(x.to(device), y.to(device), temperature_init=init_t)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return training_loop(model, optimizer, n)


def grid_search(x, y, min_v, max_v, n, use_gpu=False):
    """Grid search across T to find optimal starting temperature."""
    temps = np.linspace(min_v, max_v, n)
    device = get_device(use_gpu)
    x = x.to(device)
    y = y.to(device)
    losses = []
    for t in temps:
        model = SoftNearestNeighbours(x, y, temperature_init=t)
        model.to(device)
        with torch.no_grad():
            loss = model()
        losses.append(loss.cpu().numpy())
    return temps[np.argmin(losses)]


def get_device(use_gpu=False):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print("Using {} device: {}".format(device, torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
        print("Using {}".format(device))
    return device
