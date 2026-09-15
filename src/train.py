import torch
from IPython.display import clear_output

from src.utils import grad_norm


def train_gdmax(T, f, sample_mu, sample_nu, cost,
                 n_steps, lr_T, lr_f, K=10,
                 batch_size_x=1024, batch_size_y=1024,
                 noise_fn=None, lr_schedule=None,
                 log_every=500, callback=None, device=None):
    """GDmax: K inner Adam steps minimizing over T, then one Adam step maximizing over f.

    Args:
        noise_fn: optional (step, x) -> x, applied to each mu-batch before use.
        lr_schedule: optional step -> (lr_T, lr_f), applied to the optimizers'
            param groups every log_every steps.
        callback: optional (step, T, f), invoked every log_every steps.

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T)
    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f, maximize=True)

    loss_hist, grad_T_hist, grad_f_hist = [], [], []

    for step in range(n_steps):
        x = sample_mu(batch_size_x).to(device)
        y = sample_nu(batch_size_y).to(device)
        if noise_fn is not None:
            x = noise_fn(step, x)

        for _ in range(K):
            Tx = T(x)
            loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

            optimizer_T.zero_grad()
            loss.backward()
            optimizer_T.step()

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        T_grads = torch.autograd.grad(loss, T.parameters(), retain_graph=True)
        f_grads = torch.autograd.grad(loss, f.parameters())

        loss_hist.append(loss.item())
        grad_T_hist.append(grad_norm(T_grads).item())
        grad_f_hist.append(grad_norm(f_grads).item())

        if (step + 1) % log_every == 0:
            if lr_schedule is not None:
                new_lr_T, new_lr_f = lr_schedule(step)
                for group in optimizer_T.param_groups:
                    group["lr"] = new_lr_T
                for group in optimizer_f.param_groups:
                    group["lr"] = new_lr_f

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}


def train_extragradient(T, f, sample_mu, sample_nu, cost,
                         n_steps, lr_T, lr_f,
                         batch_size_x=1024, batch_size_y=1024,
                         lr_schedule=None, log_every=500, callback=None, device=None):
    """Extragradient: snapshot params, take a trial step, evaluate gradients at the
    trial point, restore, then apply the real update using the trial-point gradients.

    Args:
        lr_schedule: optional step -> (lr_T, lr_f), reassigns the local learning
            rates used in the manual parameter updates.
        callback: optional (step, T, f), invoked every log_every steps.

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    loss_hist, grad_T_hist, grad_f_hist = [], [], []

    for step in range(n_steps):
        x = sample_mu(batch_size_x).to(device)
        y = sample_nu(batch_size_y).to(device)

        T_old = [p.clone() for p in T.parameters()]
        f_old = [p.clone() for p in f.parameters()]

        # -------- trial step --------
        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()
        T_gradient = torch.autograd.grad(loss, T.parameters(), create_graph=True)
        f_gradient = torch.autograd.grad(loss, f.parameters())

        with torch.no_grad():
            for p, g in zip(T.parameters(), T_gradient):
                p -= lr_T * g
            for p, g in zip(f.parameters(), f_gradient):
                p += lr_f * g

        # -------- gradients at the trial point --------
        Tx2 = T(x)
        loss2 = cost(x, Tx2).mean() - f(Tx2).mean() + f(y).mean()
        T_grad_true = torch.autograd.grad(loss2, T.parameters(), create_graph=True)
        f_grad_true = torch.autograd.grad(loss2, f.parameters())

        # -------- restore and apply the real update --------
        with torch.no_grad():
            for p, p_old in zip(T.parameters(), T_old):
                p.copy_(p_old)
            for p, p_old in zip(f.parameters(), f_old):
                p.copy_(p_old)

        with torch.no_grad():
            for p, g in zip(T.parameters(), T_grad_true):
                p -= lr_T * g
            for p, g in zip(f.parameters(), f_grad_true):
                p += lr_f * g

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        T_grads = torch.autograd.grad(loss, T.parameters(), retain_graph=True)
        f_grads = torch.autograd.grad(loss, f.parameters())

        loss_hist.append(loss.item())
        grad_T_hist.append(grad_norm(T_grads).item())
        grad_f_hist.append(grad_norm(f_grads).item())

        if (step + 1) % log_every == 0:
            if lr_schedule is not None:
                lr_T, lr_f = lr_schedule(step)

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}
