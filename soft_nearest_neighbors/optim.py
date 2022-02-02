
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


def get_device(use_gpu=False):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda")
        print("Using {} device: {}".format(device, torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
        print("Using {}".format(device))
    return device
