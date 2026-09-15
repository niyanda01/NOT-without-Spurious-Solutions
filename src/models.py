import torch
import torch.nn as nn
import torch.nn.functional as F


class Transport(nn.Module):
    def __init__(self, input_dim=2, noise_dim=1, hidden_dim=64):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = torch.randn(x.shape[0], self.noise_dim, device=x.device)
        return self.net(torch.cat([x, z], dim=1))


class Critic(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ICNNCritic(nn.Module):
    """Input-Convex Neural Network critic (Amos et al., 2017)."""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.wx1 = nn.Linear(input_dim, hidden_dim)

        self.wz2_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.wx2 = nn.Linear(input_dim, hidden_dim)

        self.wz3_raw = nn.Parameter(torch.randn(1, hidden_dim))
        self.wx3 = nn.Linear(input_dim, 1)

        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = F.relu(self.wx1(x))

        # enforce non-negative weights for convexity
        wz2 = F.softplus(self.wz2_raw)
        wz3 = F.softplus(self.wz3_raw)

        z = F.relu(F.linear(z, wz2) + self.wx2(x) + self.bias2)

        # no activation on output layer to preserve convexity
        out = F.linear(z, wz3) + self.wx3(x) + self.bias3

        return 0.5 * torch.sum(x ** 2, dim=1, keepdim=True) - out / 10


class RF_Transport(nn.Module):
    """Random-Feature transport map (고정된 random features + trainable linear head)."""

    def __init__(self, input_dim=2, hidden_dim=500, output_dim=2, scale=0.3):
        super().__init__()
        self.phi = nn.Sigmoid()  # forward마다 재생성하지 않도록 __init__에서 정의

        self.register_buffer("W1", torch.randn(input_dim, hidden_dim) * scale)
        self.register_buffer("b1", torch.randn(hidden_dim) * scale)

        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * scale)
        self.b2 = nn.Parameter(torch.randn(output_dim) * scale)

    def forward(self, z):
        h1 = 2 * self.phi(z @ self.W1 + self.b1)
        return h1 @ self.W2 + self.b2


class RF_Critic(nn.Module):
    """Random-Feature critic (non-negative output weights → convex by construction)."""

    def __init__(self, input_dim=2, hidden_dim=1000, scale=0.3):
        super().__init__()
        self.phi = nn.ReLU()

        self.register_buffer("W", torch.randn(input_dim, hidden_dim) * scale)
        self.register_buffer("b", torch.randn(hidden_dim) * scale)

        self.a = nn.Parameter(torch.rand(hidden_dim, 1))

    def forward(self, x):
        h = self.phi(x @ self.W + self.b)
        return 0.5 * torch.sum(x ** 2, dim=1, keepdim=True) - h @ self.a
