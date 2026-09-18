import math
import os
import random
import matplotlib.pyplot as plt
import torch
import numpy as np


def cost(x, y):
    """Squared Euclidean transport cost c(x, y) = ||x - y||^2 / 2."""
    return ((x - y) ** 2).sum(dim=1, keepdim=True) / 2


def set_seed(seed):
    """Seed random/numpy/torch (CPU+CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_current_fig(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")


def show_mapping(T, sample_mu, sample_nu, option=False, device=None,
                 f=None, contour=False, legend=True, save_path=None):
    """Transport map T를 시각화: mu, nu, T#mu를 scatter로 표시."""
    if device is None:
        device = next(T.parameters()).device

    x = sample_mu(1000).to(device)
    outputs = torch.cat([T(x).detach().cpu() for _ in range(5)], dim=0)
    x = x.cpu()
    y = sample_nu(1000).cpu()

    plt.figure()

    if option:
        for i in range(50):
            plt.arrow(
                x[i, 0], x[i, 1],
                outputs[i, 0] - x[i, 0],
                outputs[i, 1] - x[i, 1],
                color='gray', alpha=0.5, head_width=0, length_includes_head=True,
            )

    if contour and f is not None:
        plt.gcf().set_size_inches(6, 6)
        x_min, x_max, y_min, y_max = -1.5, 1.5, -1.5, 1.5
        n = 300

        xs = np.linspace(x_min, x_max, n)
        ys = np.linspace(y_min, y_max, n)
        xx, yy = np.meshgrid(xs, ys)

        grid_torch = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            zz = f(grid_torch).cpu().numpy().reshape(n, n)

        plt.contour(xx, yy, zz, levels=20)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.scatter(x[:, 0], x[:, 1], label=r"$\mu$", s=1, alpha=0.5)
    plt.scatter(y[:, 0], y[:, 1], label=r"$\nu$", s=1, alpha=0.5)
    plt.scatter(outputs[:, 0], outputs[:, 1], label=r"$T(x,z)$", s=1, alpha=0.5)
    plt.axis('equal')
    if legend:
        plt.legend()
    if save_path is not None:
        _save_current_fig(save_path)
    plt.show()
    plt.close()


def _to_image(tensor):
    return tensor.detach().cpu().add(1).div(2).clamp(0, 1).permute(1, 2, 0)


def show_image_samples(T, x_fixed, y_fixed, method_name="T(x)", save_path=None, device=None):
    """Row grid: fixed source images, their transport T(x), fixed target images.

    x_fixed, y_fixed: (N, C, H, W) tensors in [-1, 1], shared across calls so
    every saved figure compares the same source/target images.
    """
    if device is None:
        device = next(T.parameters()).device
    T.eval()
    with torch.no_grad():
        output = T(x_fixed.to(device))
    n = x_fixed.shape[0]

    fig, axes = plt.subplots(3, n, figsize=(1.5 * n, 4.5))
    for column in range(n):
        for row, images in enumerate((x_fixed, output, y_fixed)):
            axes[row, column].imshow(_to_image(images[column]))
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("source")
    axes[1, 0].set_ylabel(method_name)
    axes[2, 0].set_ylabel("target")
    fig.tight_layout()
    if save_path is not None:
        _save_current_fig(save_path)
    plt.show()
    plt.close(fig)


def grad_norm(grads):
    """gradient tensor 리스트의 전체 L2 norm 계산."""
    return torch.sqrt(sum((g ** 2).sum() for g in grads))


def cosine_lr(step, total_steps, lr_max, lr_min=0.0):
    return lr_min + 0.5 * (lr_max - lr_min) * (
        1 + math.cos(math.pi * step / (step + total_steps))
    )


def cosine_decay(step, total_steps, lr_max):
    """Single monotone cosine decay from lr_max at step 0 to 0 at total_steps."""
    progress = min(1.0, max(0.0, step / total_steps))
    return lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))


def warmup_cosine_decay(step, warmup_steps, decay_start, decay_end, lr_max, min_frac=0.0):
    """Linear warm-up, a flat plateau at lr_max, then cosine decay to
    lr_max * min_frac, matching the handbags/shoes schedule (warm-up through
    warmup_steps, max through decay_start, cosine through decay_end, floor after)."""
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step < decay_start:
        return lr_max
    if step < decay_end:
        progress = (step - decay_start) / (decay_end - decay_start)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_max * (min_frac + (1.0 - min_frac) * cosine)
    return lr_max * min_frac


def sigma_schedule(k, K, P=2000, sigma_max=0.2, sigma_min=0.05):
    t = (P * (k // P) + 1) / K
    return (1 - t) * sigma_max + t * sigma_min


def plot_3d_function(f, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0),
                     resolution=200, device=None, cmap="viridis", save_path=None):
    if device is None:
        try:
            device = next(f.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    XY_torch = torch.tensor(
        np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32, device=device
    )

    with torch.no_grad():
        Z = f(XY_torch).detach().cpu().numpy().reshape(X.shape)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    if save_path is not None:
        _save_current_fig(save_path)
    plt.show()
    plt.close()


def ema_np(x, alpha=0.1):
    """1D 배열에 대한 지수 이동 평균(EMA)."""
    x = np.array(x, dtype=float)
    m = np.empty_like(x)
    m[0] = x[0]
    for i in range(1, len(x)):
        m[i] = alpha * x[i] + (1 - alpha) * m[i - 1]
    return m


def plot_loss(loss_hist, baseline=None, save_path=None):
    """EMA 스무딩과 함께 training loss 시각화.

    Args:
        baseline: 값이 주어지면 해당 y값에 수평 점선을 그림.
    """
    plt.figure()
    loss_plot = ema_np(loss_hist, alpha=0.05)

    plt.plot(loss_hist, label="Loss", color='tab:blue', alpha=0.5)
    plt.plot(loss_plot, label="EMA Loss", color='tab:blue')
    if baseline is not None:
        plt.axhline(y=baseline, linestyle='--', color='black')

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training Loss", fontsize=16)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        _save_current_fig(save_path)
    plt.show()
    plt.close()


def plot_grad_norm(grad_T_hist, grad_f_hist, save_path=None):
    plt.figure()
    grad_T_plot = ema_np(grad_T_hist, alpha=0.05)
    grad_f_plot = ema_np(grad_f_hist, alpha=0.05)

    plt.plot(grad_T_hist, label=r"$\| \nabla_T L \|$", color='tab:blue', alpha=0.5)
    plt.plot(grad_f_hist, label=r"$\| \nabla_V L \|$", color='tab:orange', alpha=0.5)
    plt.plot(grad_T_plot, color='tab:blue')
    plt.plot(grad_f_plot, color='tab:orange')

    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel(r"Gradient Norm", fontsize=14)
    plt.title("Gradient Norms", fontsize=16)
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        _save_current_fig(save_path)
    plt.show()
    plt.close()
