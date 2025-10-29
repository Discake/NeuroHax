import torch


class Vector:
    def __init__(self, x : torch.Tensor, y : torch.Tensor):
        self.x = x.detach().clone().requires_grad_(True)
        self.y = y.detach().clone().requires_grad_(True)

    def get_pair(self):
        return self.x, self.y