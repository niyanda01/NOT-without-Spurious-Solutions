import copy
import csv
import math
import os

import torch
from IPython.display import clear_output

from src.utils import grad_norm


def _clip_grad_list(grads, max_norm):
    """Scale a list of gradient tensors (not attached to .grad) to a max total norm."""
    total_norm = grad_norm(grads)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return list(grads)
    return [g * clip_coef for g in grads]


def _init_metrics_log(log_dir):
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as fp:
        csv.writer(fp).writerow(["step", "loss", "grad_T", "grad_f"])
    return csv_path


def _append_metrics_log(csv_path, step, loss, grad_T, grad_f):
    if csv_path is None:
        return
    with open(csv_path, "a", newline="") as fp:
        csv.writer(fp).writerow([step, loss, grad_T, grad_f])


def train_gdmax(T, f, sample_mu, sample_nu, cost,
                 n_steps, lr_T, lr_f, K=10,
                 batch_size_x=1024, batch_size_y=1024,
                 noise_fn=None, lr_schedule=None,
                 log_every=500, callback=None, device=None, log_dir=None,
                 grad_clip=None, optimizer_kwargs=None):
    """GDmax: K inner Adam steps minimizing over T, then one Adam step maximizing over f.

    Args:
        noise_fn: optional (step, x) -> x, applied to each mu-batch before use.
        lr_schedule: optional step -> (lr_T, lr_f), applied to the optimizers'
            param groups every log_every steps.
        callback: optional (step, T, f), invoked every log_every steps.
        log_dir: optional directory to write a metrics.csv (step, loss, grad_T,
            grad_f) row every log_every steps.
        grad_clip: optional max gradient norm, applied to T and f separately
            before each optimizer step.
        optimizer_kwargs: optional dict merged into both Adam constructors
            (e.g. betas, amsgrad, weight_decay).

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    metrics_csv_path = _init_metrics_log(log_dir)
    optimizer_kwargs = optimizer_kwargs or {}

    optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T, **optimizer_kwargs)
    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f, maximize=True, **optimizer_kwargs)

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
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(T.parameters(), grad_clip)
            optimizer_T.step()

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        optimizer_f.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(f.parameters(), grad_clip)
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

            _append_metrics_log(metrics_csv_path, step + 1, loss_hist[-1], grad_T_hist[-1], grad_f_hist[-1])

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}


def train_bppm(T, f, sample_mu, sample_nu, cost,
               n_steps, lr_T, lr_f, m=3,
               batch_size_x=1024, batch_size_y=1024,
               noise_fn=None, lr_schedule=None,
               log_every=500, callback=None, device=None, log_dir=None,
               grad_clip=None):
    """Binomial PPM (BPPM): approximates one proximal-point (implicit GDA) step
    to O(lr^{m+1}) using m gradient evaluations per outer step, all rebuilt
    from the current iterate every time (unlike OGDA, no dependence on the
    previous outer step's gradient). Plain gradient updates throughout (no
    Adam), matching the theorem's vector-field update exactly.

    Auxiliary points v_0 = z_t, v_1, ..., v_{m-1} are built by m-1 lookahead
    moves in the direction *opposite* the real update:
    v_{j+1} = v_j + lr * (T_grad(v_j), -f_grad(v_j)). The real update then
    applies the binomially-weighted combination
    sum_{j=0}^{m-1} (-1)^j * C(m, j+1) * grad(v_j) directly:
    T -= lr_T * combo_T, f += lr_f * combo_f. m=1 reduces to plain
    simultaneous GDA.

    Args:
        m: number of gradient evaluations per step, and the approximation
            order O(lr^{m+1}) to one proximal-point step. Default 3.
        noise_fn: optional (step, x) -> x, applied to each mu-batch before use.
        lr_schedule: optional step -> (lr_T, lr_f), applied every log_every
            steps.
        callback: optional (step, T, f), invoked every log_every steps.
        log_dir: optional directory to write a metrics.csv (step, loss, grad_T,
            grad_f) row every log_every steps.
        grad_clip: optional max gradient norm, applied to T and f separately
            at every auxiliary evaluation.

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    metrics_csv_path = _init_metrics_log(log_dir)

    T_params, f_params = list(T.parameters()), list(f.parameters())
    binom_weights = [(-1) ** j * math.comb(m, j + 1) for j in range(m)]

    loss_hist, grad_T_hist, grad_f_hist = [], [], []

    for step in range(n_steps):
        x = sample_mu(batch_size_x).to(device)
        y = sample_nu(batch_size_y).to(device)
        if noise_fn is not None:
            x = noise_fn(step, x)

        T_z = [p.detach().clone() for p in T_params]
        f_z = [p.detach().clone() for p in f_params]

        combo_T = [torch.zeros_like(p) for p in T_params]
        combo_f = [torch.zeros_like(p) for p in f_params]

        for j in range(m):
            Tx = T(x)
            loss_j = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

            T_grads = torch.autograd.grad(loss_j, T_params, retain_graph=True)
            f_grads = torch.autograd.grad(loss_j, f_params)
            if grad_clip is not None:
                T_grads = _clip_grad_list(T_grads, grad_clip)
                f_grads = _clip_grad_list(f_grads, grad_clip)

            w = binom_weights[j]
            for c, g in zip(combo_T, T_grads):
                c.add_(g, alpha=w)
            for c, g in zip(combo_f, f_grads):
                c.add_(g, alpha=w)

            if j < m - 1:
                with torch.no_grad():
                    for p, g in zip(T_params, T_grads):
                        p.add_(g, alpha=lr_T)
                    for p, g in zip(f_params, f_grads):
                        p.add_(g, alpha=-lr_f)

        # restore params to z_t, then apply the real (plain) update on the combo
        with torch.no_grad():
            for p, v in zip(T_params, T_z):
                p.copy_(v)
            for p, v in zip(f_params, f_z):
                p.copy_(v)
            for p, c in zip(T_params, combo_T):
                p.add_(c, alpha=-lr_T)
            for p, c in zip(f_params, combo_f):
                p.add_(c, alpha=lr_f)

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        T_grads_log = torch.autograd.grad(loss, T_params, retain_graph=True)
        f_grads_log = torch.autograd.grad(loss, f_params)

        loss_hist.append(loss.item())
        grad_T_hist.append(grad_norm(T_grads_log).item())
        grad_f_hist.append(grad_norm(f_grads_log).item())

        if (step + 1) % log_every == 0:
            if lr_schedule is not None:
                lr_T, lr_f = lr_schedule(step)

            _append_metrics_log(metrics_csv_path, step + 1, loss_hist[-1], grad_T_hist[-1], grad_f_hist[-1])

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}


def train_ogda(T, f, sample_mu, sample_nu, cost,
               n_steps, lr_T, lr_f,
               batch_size_x=1024, batch_size_y=1024,
               noise_fn=None, lr_schedule=None,
               log_every=500, callback=None, device=None, log_dir=None,
               grad_clip=None, optimizer_kwargs=None):
    """Optimistic GDA (OAdam): each Adam step uses the extrapolated gradient
    2*g_t - g_{t-1} in place of the raw gradient g_t, so the update
    anticipates the opponent's next move instead of reacting to the last one.
    The very first step falls back to the raw gradient (g_{-1} := g_0).

    Args:
        noise_fn: optional (step, x) -> x, applied to each mu-batch before use.
        lr_schedule: optional step -> (lr_T, lr_f), applied to the optimizers'
            param groups every log_every steps.
        callback: optional (step, T, f), invoked every log_every steps.
        log_dir: optional directory to write a metrics.csv (step, loss, grad_T,
            grad_f) row every log_every steps.
        grad_clip: optional max gradient norm, applied to T and f separately
            before each optimizer step (on the raw, non-extrapolated gradient).
        optimizer_kwargs: optional dict merged into both Adam constructors
            (e.g. betas, amsgrad, weight_decay).

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    metrics_csv_path = _init_metrics_log(log_dir)
    optimizer_kwargs = optimizer_kwargs or {}

    optimizer_T = torch.optim.Adam(T.parameters(), lr=lr_T, **optimizer_kwargs)
    optimizer_f = torch.optim.Adam(f.parameters(), lr=lr_f, maximize=True, **optimizer_kwargs)

    T_params, f_params = list(T.parameters()), list(f.parameters())
    T_prev_grads, f_prev_grads = None, None

    loss_hist, grad_T_hist, grad_f_hist = [], [], []

    for step in range(n_steps):
        x = sample_mu(batch_size_x).to(device)
        y = sample_nu(batch_size_y).to(device)
        if noise_fn is not None:
            x = noise_fn(step, x)

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        optimizer_T.zero_grad()
        optimizer_f.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(T_params, grad_clip)
            torch.nn.utils.clip_grad_norm_(f_params, grad_clip)

        T_grads = [p.grad.detach().clone() for p in T_params]
        f_grads = [p.grad.detach().clone() for p in f_params]

        if T_prev_grads is None:
            T_prev_grads = [g.clone() for g in T_grads]
        if f_prev_grads is None:
            f_prev_grads = [g.clone() for g in f_grads]

        for p, g, g_prev in zip(T_params, T_grads, T_prev_grads):
            p.grad = 2 * g - g_prev
        for p, g, g_prev in zip(f_params, f_grads, f_prev_grads):
            p.grad = 2 * g - g_prev

        optimizer_T.step()
        optimizer_f.step()

        T_prev_grads, f_prev_grads = T_grads, f_grads

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        T_grads_log = torch.autograd.grad(loss, T_params, retain_graph=True)
        f_grads_log = torch.autograd.grad(loss, f_params)

        loss_hist.append(loss.item())
        grad_T_hist.append(grad_norm(T_grads_log).item())
        grad_f_hist.append(grad_norm(f_grads_log).item())

        if (step + 1) % log_every == 0:
            if lr_schedule is not None:
                new_lr_T, new_lr_f = lr_schedule(step)
                for group in optimizer_T.param_groups:
                    group["lr"] = new_lr_T
                for group in optimizer_f.param_groups:
                    group["lr"] = new_lr_f

            _append_metrics_log(metrics_csv_path, step + 1, loss_hist[-1], grad_T_hist[-1], grad_f_hist[-1])

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}


def train_extragradient(T, f, sample_mu, sample_nu, cost,
                         n_steps, lr_T, lr_f,
                         batch_size_x=1024, batch_size_y=1024,
                         lr_schedule=None, log_every=500, callback=None, device=None,
                         log_dir=None, grad_clip=None, use_adam=False, optimizer_kwargs=None):
    """Extragradient: snapshot params, take a trial gradient-descent-ascent
    step, evaluate gradients at the trial point, restore, then apply the real
    step using the trial-point gradients.

    By default (use_adam=False) both the trial and real steps are plain
    gradient steps, matching the original construction. Set use_adam=True to
    instead take both steps through Adam optimizers (T minimizing, f
    maximizing), snapshotting and restoring their state around the trial step.

    Args:
        lr_schedule: optional step -> (lr_T, lr_f), applied every log_every
            steps (to the optimizers' param groups when use_adam=True).
        callback: optional (step, T, f), invoked every log_every steps.
        log_dir: optional directory to write a metrics.csv (step, loss, grad_T,
            grad_f) row every log_every steps.
        grad_clip: optional max gradient norm, applied to T and f separately
            at both the trial and real gradient evaluations.
        use_adam: if True, use Adam instead of plain gradient steps.
        optimizer_kwargs: optional dict merged into both Adam constructors
            when use_adam=True (e.g. betas, amsgrad, weight_decay).

    Returns:
        {"loss": [...], "grad_T": [...], "grad_f": [...]}
    """
    if device is None:
        device = next(T.parameters()).device

    metrics_csv_path = _init_metrics_log(log_dir)

    T_params, f_params = list(T.parameters()), list(f.parameters())

    if use_adam:
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer_T = torch.optim.Adam(T_params, lr=lr_T, **optimizer_kwargs)
        optimizer_f = torch.optim.Adam(f_params, lr=lr_f, maximize=True, **optimizer_kwargs)

    loss_hist, grad_T_hist, grad_f_hist = [], [], []

    for step in range(n_steps):
        x = sample_mu(batch_size_x).to(device)
        y = sample_nu(batch_size_y).to(device)

        T_old = [p.detach().clone() for p in T_params]
        f_old = [p.detach().clone() for p in f_params]
        if use_adam:
            T_opt_state = copy.deepcopy(optimizer_T.state_dict())
            f_opt_state = copy.deepcopy(optimizer_f.state_dict())

        # -------- trial step --------
        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()
        T_grads = torch.autograd.grad(loss, T_params, retain_graph=True)
        f_grads = torch.autograd.grad(loss, f_params)
        if grad_clip is not None:
            T_grads = _clip_grad_list(T_grads, grad_clip)
            f_grads = _clip_grad_list(f_grads, grad_clip)

        if use_adam:
            for p, g in zip(T_params, T_grads):
                p.grad = g
            for p, g in zip(f_params, f_grads):
                p.grad = g
            optimizer_T.step()
            optimizer_f.step()
        else:
            with torch.no_grad():
                for p, g in zip(T_params, T_grads):
                    p.add_(g, alpha=-lr_T)
                for p, g in zip(f_params, f_grads):
                    p.add_(g, alpha=lr_f)

        # -------- gradients at the trial point --------
        Tx2 = T(x)
        loss2 = cost(x, Tx2).mean() - f(Tx2).mean() + f(y).mean()
        T_grads2 = torch.autograd.grad(loss2, T_params, retain_graph=True)
        f_grads2 = torch.autograd.grad(loss2, f_params)
        if grad_clip is not None:
            T_grads2 = _clip_grad_list(T_grads2, grad_clip)
            f_grads2 = _clip_grad_list(f_grads2, grad_clip)

        # -------- restore params (and Adam state), apply the real step --------
        with torch.no_grad():
            for p, p_old in zip(T_params, T_old):
                p.copy_(p_old)
            for p, p_old in zip(f_params, f_old):
                p.copy_(p_old)

        if use_adam:
            optimizer_T.load_state_dict(T_opt_state)
            optimizer_f.load_state_dict(f_opt_state)
            for p, g in zip(T_params, T_grads2):
                p.grad = g
            for p, g in zip(f_params, f_grads2):
                p.grad = g
            optimizer_T.step()
            optimizer_f.step()
        else:
            with torch.no_grad():
                for p, g in zip(T_params, T_grads2):
                    p.add_(g, alpha=-lr_T)
                for p, g in zip(f_params, f_grads2):
                    p.add_(g, alpha=lr_f)

        Tx = T(x)
        loss = cost(x, Tx).mean() - f(Tx).mean() + f(y).mean()

        T_grads_log = torch.autograd.grad(loss, T_params, retain_graph=True)
        f_grads_log = torch.autograd.grad(loss, f_params)

        loss_hist.append(loss.item())
        grad_T_hist.append(grad_norm(T_grads_log).item())
        grad_f_hist.append(grad_norm(f_grads_log).item())

        if (step + 1) % log_every == 0:
            if lr_schedule is not None:
                lr_T, lr_f = lr_schedule(step)
                if use_adam:
                    for group in optimizer_T.param_groups:
                        group["lr"] = lr_T
                    for group in optimizer_f.param_groups:
                        group["lr"] = lr_f

            _append_metrics_log(metrics_csv_path, step + 1, loss_hist[-1], grad_T_hist[-1], grad_f_hist[-1])

            clear_output(wait=True)
            print(f"step {step + 1}, loss {loss.item():.4f}")
            if callback is not None:
                callback(step, T, f)

    return {"loss": loss_hist, "grad_T": grad_T_hist, "grad_f": grad_f_hist}
