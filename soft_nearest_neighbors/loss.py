
import torch


class SoftNearestNeighbours(torch.nn.Module):
    def __init__(self, x, y, temperature_init: float = 0.1, eps: float = 1e-8, raise_on_inf=False, raise_on_single_point_for_class=False):
        super(SoftNearestNeighbours, self).__init__()
        weights = torch.zeros(1) + 1/temperature_init
        self.weights = torch.nn.Parameter(weights)
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        self.distances = torch.cdist(x, x, p=2)
        self.eps = eps
        self.raise_on_inf = raise_on_inf
        self.raise_on_single_point_for_class = raise_on_single_point_for_class

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
        ignore_idx = []
        for i in range(self.n):
            top = exponents[i, self.get_bool_mask(i, True)]
            bot = exponents[i, self.get_bool_mask(i, False)]
            fracs[i] = torch.log(top.sum()) - torch.log(bot.sum()+self.eps)
            if len(top) == 0:
                if self.raise_on_single_point_for_class:
                    raise ValueError("No other points with the same label in batch")
                else:
                    ignore_idx.append(i)
            elif fracs[i].isnan().item():
                raise ValueError("Nan detected in loss calculation")
            elif fracs[i].isinf().item() and self.raise_on_inf:
                raise ValueError("inf detected in loss calculation, if optimising try reducing the lr")
        ignore_mask = torch.ones(self.n, dtype=torch.bool)
        for i in ignore_idx:
            ignore_mask[i] = False
        fracs = fracs[ignore_mask]
        loss = (-1 / len(fracs)) * fracs.sum()
        return loss
