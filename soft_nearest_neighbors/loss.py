
import torch


class SoftNearestNeighbours(torch.nn.Module):
    def __init__(self, x, y, temperature_init: float = 0.1, eps: float = 1e-8):
        super(SoftNearestNeighbours, self).__init__()
        weights = torch.zeros(1) + 1/temperature_init
        self.weights = torch.nn.Parameter(weights)
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        self.distances = torch.cdist(x, x, p=2)
        self.eps = eps

    def get_bool_mask(self, idx, class_subset=False):
        if class_subset:
            mask = self.y == self.y[idx]
        else:
            mask = torch.ones(self.n, dtype=bool)
        mask[idx] = False
        return mask

    def forward(self):
        fracs = torch.zeros(self.n)
        exponents = torch.exp(-self.distances * self.weights[0])
        for i in range(self.n):
            top = exponents[i, self.get_bool_mask(i, True)]
            bot = exponents[i, self.get_bool_mask(i, False)]
            fracs[i] = torch.log(top.sum()/(bot.sum() + self.eps))
            if fracs[i].isnan().item():
                raise ValueError("Nan detected in loss calculation")
        loss = (-1 / self.n) * fracs.sum()
        return loss
