import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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


class ResNetTransport(nn.Module):
    """Stochastic transport map: same noise injection as Transport, but with
    residual MLP blocks (x = x + block(x)) instead of a plain feedforward net."""

    def __init__(self, input_dim=2, noise_dim=1, hidden_dim=64, n_blocks=3):
        super().__init__()
        self.noise_dim = noise_dim
        self.input_proj = nn.Linear(input_dim + noise_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = torch.randn(x.shape[0], self.noise_dim, device=x.device)
        hidden = F.relu(self.input_proj(torch.cat([x, z], dim=1)))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_proj(hidden)


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


def _groups(channels):
    """Largest group count in (16, 8, 4, 2, 1) that evenly divides channels."""
    for groups in (16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.block(inputs)


class TransportUNet(nn.Module):
    """Deterministic two-level U-Net transport map, tanh-bounded to [-1, 1]."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.enc1 = ConvGNAct(3, base_channels)
        self.down1 = nn.Conv2d(base_channels, 2 * base_channels, 4, stride=2, padding=1)
        self.enc2 = ConvGNAct(2 * base_channels, 2 * base_channels)
        self.down2 = nn.Conv2d(2 * base_channels, 4 * base_channels, 4, stride=2, padding=1)
        self.bottleneck = ConvGNAct(4 * base_channels, 4 * base_channels)
        self.up1_conv = nn.Conv2d(4 * base_channels, 2 * base_channels, 3, padding=1)
        self.dec1 = ConvGNAct(4 * base_channels, 2 * base_channels)
        self.up2_conv = nn.Conv2d(2 * base_channels, base_channels, 3, padding=1)
        self.dec2 = ConvGNAct(2 * base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, 3, 1)

    def forward(self, inputs):
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.down1(enc1))
        hidden = self.bottleneck(self.down2(enc2))
        hidden = F.interpolate(hidden, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up1_conv(hidden)
        hidden = self.dec1(torch.cat((hidden, enc2), dim=1))
        hidden = F.interpolate(hidden, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.up2_conv(hidden)
        hidden = self.dec2(torch.cat((hidden, enc1), dim=1))
        return torch.tanh(self.output(hidden))


class ResidualDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, inputs):
        main = self.main(inputs)
        skip = self.skip(F.avg_pool2d(inputs, 2))
        return (main + skip) / (2.0 ** 0.5)


class PotentialResNet(nn.Module):
    """Spectral-normalized ResNet scalar potential V for image inputs."""

    def __init__(self, base_channels=32):
        super().__init__()
        self.stem = spectral_norm(nn.Conv2d(3, base_channels, 3, padding=1))
        self.blocks = nn.Sequential(
            ResidualDown(base_channels, 2 * base_channels),
            ResidualDown(2 * base_channels, 4 * base_channels),
            ResidualDown(4 * base_channels, 8 * base_channels),
        )
        self.head = spectral_norm(nn.Linear(8 * base_channels, 1))

    def forward(self, inputs):
        features = self.blocks(self.stem(inputs))
        features = F.leaky_relu(features, 0.2)
        features = features.mean(dim=(2, 3))
        return self.head(features)


def warmup_spectral_norm(V, image_size, n_iters=10, device=None):
    """Run a few dummy forward passes so every spectral-norm power-iteration
    buffer in V is initialized before training starts."""
    if device is None:
        device = next(V.parameters()).device
    was_training = V.training
    V.train()
    dummy = torch.zeros(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(n_iters):
            V(dummy)
    V.train(was_training)


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
