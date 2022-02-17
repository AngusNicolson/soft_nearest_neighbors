import numpy as np
import torch

from soft_nearest_neighbors.loss import SoftNearestNeighbours


def training_loop(model, optimizer, n=20, tol=1e-4, min_iter=6):
    if min_iter <= 2:
        raise ValueError("Minimum no. iterations must be greater than 2")
    losses = []
    flags = {"converged": False, "increased": False, "finished": False}
    temps = []
    for i in range(n):
        temps.append(1/model.weights.detach().cpu().numpy()[0])
        loss = model()
        loss.backward()
        optimizer.step()
        model.weights.data = model.weights.data.clamp(min=1e-5)
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        del loss
        if np.allclose(losses[-1], losses[-min_iter:], rtol=0, atol=tol) and len(losses) >= min_iter:
            print(f"Loss converged to {losses[-1]:.4f} in {i+1} iterations with T {temps[-1]:.3f}")
            flags["converged"] = True
            break
        elif not is_converging(losses, horizon=min_iter):
            flags["increased"] = True
            print(f"Loss increased! {losses[-1]:.4f} in {i+1} iterations with T {temps[-1]:.3f}. Use a smaller lr.")
            break
        elif i == n - 1:
            flags["finished"] = True
            print(f"Loss failed to converge with {n} iterations.  Try using a higher lr. Final values: {losses[-1]:.4f}, T {temps[-1]:.3f}")

    return losses, temps, flags


def is_converging(losses, horizon=5):
    """Check if the loss is reducing."""
    if len(losses) <= horizon:
        return True
    else:
        losses = np.array(losses[:horizon])
        running_mean = np.convolve(losses, np.ones(2)/2, mode="valid")
        deltas = losses[1:] - running_mean
        if (deltas <= 0).all():
            return True
        else:
            return False


def grid_searches(x, y, lows, highs, ns=20, use_gpu=False):
    """Run multiple grid searches to find the optimal starting temperature"""
    if len(lows) != len(highs):
        raise ValueError("Must provide same number same number of low and high values")
    if type(ns) == int:
        ns = [n for n in range(len(lows))]
    losses = []
    temps = []
    for low, high, n in zip(lows, highs, ns):
        grid_loss, init_t = grid_search(x, y, low, high, n, use_gpu=use_gpu)
        losses.append(grid_loss)
        temps.append(init_t)

    minimum_idx = np.argmin(losses)
    print(f"Grid search complete. Initial values: T: {temps[minimum_idx]:.2f}, Loss: {losses[minimum_idx]:.4f}.")
    return temps[minimum_idx], losses[minimum_idx]


def get_loss(x, y, lr=0.1, n=20, init_t=0.1, use_gpu=False, tol=1e-4, min_iter=6):
    device = get_device(use_gpu)
    model = SoftNearestNeighbours(x.to(device), y.to(device), temperature_init=init_t, raise_on_inf=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return training_loop(model, optimizer, n, tol=tol, min_iter=min_iter)


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
    min_idx = np.argmin(losses)
    return losses[min_idx], temps[min_idx]


def get_device(use_gpu=False, log=False):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        if log:
            print("Using {} device: {}".format(device, torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
        if log:
            print("Using {}".format(device))
    return device
