import torch


class OptimisticAdam(torch.optim.Optimizer):
    """Optimistic Adam (Daskalakis et al., 2018, "Training GANs with Optimism").

    Optimism is applied to Adam's normalized step rather than to the raw
    gradient: with u_t = m_hat_t / (sqrt(v_hat_t) + eps), the parameters move by
    -lr * (2 u_t - u_{t-1}). The extra -u_{t-1} term anticipates the opponent's
    next move and damps the rotation that plain simultaneous Adam/GDA shows in
    min-max games.

    Hyperparameters mirror torch.optim.Adam (betas, eps, weight_decay as an L2
    term added to the gradient, amsgrad, maximize), so it is a drop-in
    replacement that changes only the update rule.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, amsgrad=False, maximize=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, maximize=maximize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = -p.grad if group["maximize"] else p.grad
                if group["weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["prev_update"] = torch.zeros_like(p)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group["amsgrad"]:
                    torch.maximum(state["max_exp_avg_sq"], exp_avg_sq,
                                  out=state["max_exp_avg_sq"])
                    second_moment = state["max_exp_avg_sq"]
                else:
                    second_moment = exp_avg_sq

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                denom = (second_moment / bias_correction2).sqrt_().add_(group["eps"])
                update = (exp_avg / bias_correction1).div_(denom)

                p.add_(2 * update - state["prev_update"], alpha=-group["lr"])
                state["prev_update"] = update

        return loss
