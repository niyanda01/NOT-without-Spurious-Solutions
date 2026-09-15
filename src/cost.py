import torch


def _add_noise(x, sigma):
    """sigma > 0이면 Gaussian noise를 추가하고, 아니면 x를 그대로 반환."""
    if sigma > 0.0:
        return x + sigma * torch.randn_like(x)
    return x


def compute_transport_cost(T, sample_mu, sigma=0.0, n_samples=1024, device=None):
    """E[||T(x + noise) - x||^2] 계산.

    Args:
        sigma: 입력 노이즈 표준편차 (0 = 노이즈 없음).
    """
    if device is None:
        device = next(T.parameters()).device

    x = sample_mu(n_samples).to(device)
    x_in = _add_noise(x, sigma)

    with torch.no_grad():
        Tx = T(x_in)

    cost = ((Tx - x) ** 2).sum(dim=1).mean()
    return cost.item()


# backward-compatible alias
def compute_transport_cost_noise(T, sample_mu, sigma=0.05, n_samples=1024, device=None):
    return compute_transport_cost(T, sample_mu, sigma=sigma, n_samples=n_samples, device=device)


def sinkhorn_w2(x, y, epsilon=0.05, n_iter=1000):
    """x, y의 경험 분포 사이의 Sinkhorn W2 근사값 계산."""
    n, m = x.shape[0], y.shape[0]

    a = torch.full((n,), 1.0 / n, device=x.device)
    b = torch.full((m,), 1.0 / m, device=y.device)

    # cost matrix (squared Euclidean)
    C = torch.sum((x.unsqueeze(1) - y.unsqueeze(0)) ** 2, dim=2)  # (n, m)

    # log-domain Sinkhorn iterations
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(n_iter):
        f = -epsilon * torch.logsumexp((g.unsqueeze(0) - C) / epsilon, dim=1) + epsilon * torch.log(a)
        g = -epsilon * torch.logsumexp((f.unsqueeze(1) - C) / epsilon, dim=0) + epsilon * torch.log(b)

    log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
    w2 = torch.sum(torch.exp(log_P) * C)
    return w2


def compute_w2_sinkhorn(T, sample_mu, sample_nu, sigma=0.0,
                        n_samples=512, epsilon=0.05, device=None):
    """T#mu와 nu 사이의 Sinkhorn-W2 계산.

    Args:
        sigma: mu 샘플에 추가할 입력 노이즈 표준편차 (0 = 노이즈 없음).
    """
    if device is None:
        device = next(T.parameters()).device

    x = sample_mu(n_samples).to(device)
    x_in = _add_noise(x, sigma)
    y_true = sample_nu(n_samples).to(device)

    with torch.no_grad():
        y_pred = T(x_in)

    return sinkhorn_w2(y_pred, y_true, epsilon=epsilon).item()


# backward-compatible alias
def compute_w2_sinkhorn_noise(T, sample_mu, sample_nu, sigma=0.05,
                              n_samples=512, epsilon=0.05, device=None):
    return compute_w2_sinkhorn(T, sample_mu, sample_nu, sigma=sigma,
                               n_samples=n_samples, epsilon=epsilon, device=device)
