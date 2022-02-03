import numpy as np
import torch

from soft_nearest_neighbors.loss import SoftNearestNeighbours


def training_loop(model, optimizer, n=20, tol=1e-4):
    losses = []
    converged = False
    temps = [model.weights.detach().cpu().numpy()[0]]
    for i in range(n):
        loss = model()
        loss.backward()
        optimizer.step()
        model.weights.data = model.weights.data.clamp(min=1e-5)
        optimizer.zero_grad()
        losses.append(loss.detach().cpu().numpy())
        temps.append(model.weights.detach().cpu().numpy()[0])
        del loss
        if not is_converging(losses):
            print("Loss increased! Use a smaller lr.")
            break
        elif np.allclose(losses[-1], losses[-5:], rtol=0, atol=tol) and len(losses) > 5:
            print(f"Loss converged to {losses[-1]:.4f} in {i+1} iterations")
            converged = True
            break
    return losses, temps, converged


def is_converging(losses, horizon=5):
    """Check if the loss is reducing."""
    if len(losses) < horizon:
        return True
    else:
        losses = np.array(losses[:5])
        running_mean = np.convolve(losses, np.ones(2)/2, mode="valid")
        deltas = losses[1:] - running_mean
        if (deltas <= 0).all():
            return True
        else:
            return False


def get_loss(x, y, lr=0.1, n=20, init_t=0.1, use_gpu=False, tol=1e-4):
    device = get_device(use_gpu)
    model = SoftNearestNeighbours(x.to(device), y.to(device), temperature_init=init_t, raise_on_inf=True)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return training_loop(model, optimizer, n, tol=tol)


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


def get_device(use_gpu=False):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print("Using {} device: {}".format(device, torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
        print("Using {}".format(device))
    return device
