import csv
import math
import os

import torch
from scipy.optimize import linear_sum_assignment


def exact_w2_per_dimension(first, second, chunk_size=64):
    """W2^2 / d between two uniform empirical measures of equal size.

    Rows of first/second are flattened samples already divided by sqrt(d).
    Equal uniform weights reduce discrete OT to a linear assignment problem;
    the cost matrix is built in row chunks to avoid a large cdist temporary."""
    if len(first) != len(second):
        raise ValueError("Exact assignment requires equal support sizes")
    cost = torch.cat([
        torch.cdist(first[start:start + chunk_size], second).square().cpu()
        for start in range(0, len(first), chunk_size)
    ], dim=0)
    rows, columns = linear_sum_assignment(cost.numpy())
    return float(cost[rows, columns].mean())


class OTEvaluator:
    """Tracks transport quality on fixed source/target supports:

        D_cost   = |W2^2(mu, nu)/d - E||x - T(x)||^2/d|
        D_target = W2^2(T#mu, nu)/d

    Supports are max_samples images drawn once (seeded) from each dataset and
    kept in memory; evaluate(T, n) uses their first n images, so a smaller
    periodic evaluation is a prefix of the final one."""

    def __init__(self, source, target, max_samples, seed, device, batch_size=64):
        if max_samples > min(len(source), len(target)):
            raise ValueError(
                f"OT support {max_samples} exceeds dataset sizes "
                f"({len(source)}, {len(target)})")
        generator = torch.Generator().manual_seed(seed)
        source_ids = torch.randperm(len(source), generator=generator)[:max_samples]
        target_ids = torch.randperm(len(target), generator=generator)[:max_samples]
        self.source = torch.stack([source[int(i)][0] for i in source_ids])
        self.target = torch.stack([target[int(i)][0] for i in target_ids])
        self.device = device
        self.batch_size = batch_size
        self.dimension = self.source[0].numel()
        self._ot_costs = {}

    def _flatten(self, images):
        return images.flatten(1) / math.sqrt(self.dimension)

    @torch.no_grad()
    def evaluate(self, T, n_samples):
        was_training = T.training
        T.eval()
        try:
            source = self.source[:n_samples].to(self.device)
            pushed = torch.cat([
                T(source[start:start + self.batch_size])
                for start in range(0, n_samples, self.batch_size)
            ]).clamp(-1.0, 1.0)
        finally:
            T.train(was_training)
        if not torch.isfinite(pushed).all():
            raise FloatingPointError("Non-finite transport output during OT evaluation.")

        source = self._flatten(source)
        pushed = self._flatten(pushed)
        target = self._flatten(self.target[:n_samples].to(self.device))

        if n_samples not in self._ot_costs:
            self._ot_costs[n_samples] = exact_w2_per_dimension(source, target)
        ot_cost = self._ot_costs[n_samples]
        learned_cost = (source - pushed).square().sum(1).mean().item()
        unique_targets = int(torch.cdist(pushed, target).argmin(dim=1).unique().numel())
        output_pairwise = torch.pdist(pushed).square().mean().item()
        target_pairwise = torch.pdist(target).square().mean().item()
        return {
            "ot_sample_count": n_samples,
            "OT_cost": ot_cost,
            "learned_transport_cost": learned_cost,
            "D_cost": abs(ot_cost - learned_cost),
            "D_target": exact_w2_per_dimension(pushed, target),
            "unique_target_count": unique_targets,
            "unique_target_coverage": unique_targets / n_samples,
            "diversity_ratio": output_pairwise / max(target_pairwise, 1e-12),
        }


def log_ot_metrics(out_dir, step, metrics):
    """Append one OTEvaluator.evaluate() result as a row of
    out_dir/ot_metrics.csv (header on first write) and print a summary."""
    path = os.path.join(out_dir, "ot_metrics.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["step", *metrics])
        if new_file:
            writer.writeheader()
        writer.writerow({"step": step, **metrics})
    print(f"[OT] step {step:,} | D_cost={metrics['D_cost']:.6f} "
          f"D_target={metrics['D_target']:.6f} "
          f"coverage={metrics['unique_target_count']}/{metrics['ot_sample_count']} "
          f"diversity_ratio={metrics['diversity_ratio']:.3f}", flush=True)
