import torch
from torch import nn


class DotProduct(nn.Module):
    def __init__(self, dim):
        super(DotProduct, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_pairwise):
        if is_pairwise:
            return (x * y).sum(-1)
        else:
            return x.mm(y.transpose(0, 1))


class Bilinear(nn.Module):
    def __init__(self, dim):
        super(Bilinear, self).__init__()
        self.W = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_pairwise):
        if is_pairwise:
            return (self.W(x) * y).sum(-1)
        else:
            return self.W(x).mm(y.transpose(0, 1))


class DiagonalBilinear(nn.Module):
    def __init__(self, dim):
        super(DiagonalBilinear, self).__init__()
        self.W = nn.Parameter(torch.Tensor(dim))
        nn.init.uniform_(self.W)

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_pairwise):
        if is_pairwise:
            return (x * self.W * y).sum(-1)
        else:
            return (x * self.W).mm(y.transpose(0, 1))


class Additive(nn.Module):
    def __init__(self, dim, activation='ReLU', bias=False):
        super(Additive, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=bias),
            getattr(nn, activation)(),
            nn.Linear(dim, 1, bias=bias),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, is_pairwise):
        if is_pairwise:
            return self.mlp(torch.cat([x, y], dim=-1))
        else:
            out = torch.cat([x.unsqueeze(1).expand(-1, y.size(0), -1),
                             y.unsqueeze(0).expand(x.size(0), -1, -1)], dim=-1)
            return self.mlp(out)
