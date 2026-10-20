"""NOT training: red handbags -> blue shoes, as a batch script.

Same experiment as notebook `8. NOT_HandbagsToShoes.ipynb`: the white
product-photo background is masked out by default so only the handbag's R
channel and the shoe's B channel remain. Following the paper's original
training code (NOT_training_strong.py), the LR schedule (warm-up from 0.1x,
hold, cosine to 0.05x) is applied every step, EG steps through Adam, GDmax
draws fresh batches for every T/f step, and D_cost / D_target are evaluated
on fixed supports.

Each run owns its --out-dir: its settings are saved to config.json, leftovers
of an earlier unfinished run there are cleared first, and a finished run
(final.pt) is only replaced with --overwrite.

Preprocessed images are cached under data/cache/ (built on first use; run
scripts/preprocess_product_data.py beforehand to build it once up front).

Usage (from the repo root):
    python scripts/run_handbags_to_shoes.py --method eg
    python scripts/run_handbags_to_shoes.py --method gdmax --keep-background
"""
import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import torch
import torch.nn.functional as F

from src.datasets import LoaderSampler, build_handbags_to_shoes
from src.ot_metrics import OTEvaluator, log_ot_metrics
from src.models import PotentialResNet, TransportUNet, warmup_spectral_norm
from src.train import train_extragradient, train_gdmax
from src.utils import (plot_grad_norm, plot_loss, prepare_run_dir, set_seed,
                       show_image_samples, warmup_cosine_decay)

ADAM_KWARGS = dict(betas=(0.0, 0.9), weight_decay=1e-10, amsgrad=True)
# Notebook 8 defaults per method: (lr_T, lr_f).
DEFAULT_LRS = {"gdmax": (5e-5, 5e-5), "eg": (1e-4, 1e-4)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=["gdmax", "eg"], required=True)
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--out-dir", default=None,
                        help="default: outputs/handbags_to_shoes[_whitebg]/<method>")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a finished run already in --out-dir")
    parser.add_argument("--keep-background", action="store_true",
                        help="old behavior: unused channels = 1 (white background)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=160_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--decay-start", type=int, default=20_000)
    parser.add_argument("--decay-end", type=int, default=150_000)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup-start-frac", type=float, default=0.1)
    parser.add_argument("--eg-plain-gd", action="store_true",
                        help="EG with plain gradient steps instead of Adam")
    parser.add_argument("--lr-T", type=float, default=None)
    parser.add_argument("--lr-f", type=float, default=None)
    parser.add_argument("--K", type=int, default=5, help="GDmax inner T steps")
    parser.add_argument("--gdmax-reuse-batch", action="store_true",
                        help="GDmax reuses one batch pair for all K+1 steps "
                             "instead of resampling every step")
    parser.add_argument("--no-cache", action="store_true",
                        help="preprocess images on the fly instead of data/cache/")
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--min-border-white", type=float, default=0.5,
                        help="drop photos whose border is less than this fraction "
                             "near-white (non-white backgrounds); 0 keeps all")
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ot-eval-every", type=int, default=10_000,
                        help="D_cost/D_target interval (multiple of --log-every)")
    parser.add_argument("--ot-eval-samples", type=int, default=1_000)
    parser.add_argument("--final-ot-eval-samples", type=int, default=2_000)
    args = parser.parse_args()
    if args.ot_eval_every % args.log_every:
        parser.error("--ot-eval-every must be a multiple of --log-every")
    if args.final_ot_eval_samples < args.ot_eval_samples:
        parser.error("--final-ot-eval-samples must be >= --ot-eval-samples")
    return args


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr_T = args.lr_T if args.lr_T is not None else DEFAULT_LRS[args.method][0]
    lr_f = args.lr_f if args.lr_f is not None else DEFAULT_LRS[args.method][1]

    exp_name = "handbags_to_shoes_whitebg" if args.keep_background else "handbags_to_shoes"
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "outputs", exp_name, args.method)
    prepare_run_dir(out_dir, overwrite=args.overwrite)

    source_data, target_data = build_handbags_to_shoes(
        args.data_root, img_size=args.img_size,
        remove_background=not args.keep_background,
        use_cache=not args.no_cache, num_workers=args.cache_workers,
        min_border_white=args.min_border_white)
    print(f"handbag photos: {len(source_data):,}")
    print(f"shoe photos: {len(target_data):,}")

    config = {**vars(args), "out_dir": out_dir, "lr_T": lr_T, "lr_f": lr_f,
              "n_source": len(source_data), "n_target": len(target_data),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    with open(os.path.join(out_dir, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    source_sampler = LoaderSampler(source_data, args.batch_size, args.seed, device)
    target_sampler = LoaderSampler(target_data, args.batch_size, args.seed + 1, device)
    # Fixed images so every saved figure compares the same source/target samples.
    x_fixed = LoaderSampler(source_data, 10, args.seed + 10_000, device).sample()
    y_fixed = LoaderSampler(target_data, 10, args.seed + 10_001, device).sample()

    def lr_schedule(step):
        return tuple(
            warmup_cosine_decay(step, args.warmup_steps, args.decay_start,
                                args.decay_end, lr, args.min_lr_frac,
                                args.warmup_start_frac)
            for lr in (lr_T, lr_f))

    ot_evaluator = OTEvaluator(source_data, target_data, args.final_ot_eval_samples,
                               seed=args.seed + 60_000, device=device)
    best_d_target = math.inf

    def callback(step, T, f):
        nonlocal best_d_target
        show_image_samples(
            T, x_fixed, y_fixed, method_name=args.method,
            save_path=os.path.join(out_dir, f"samples_step_{step + 1:06d}.png"))
        torch.save({"step": step, "T": T.state_dict(), "f": f.state_dict()},
                   os.path.join(out_dir, "checkpoint.pt"))
        # The last step is evaluated on the larger final support below.
        if (step + 1) % args.ot_eval_every or step + 1 == args.n_steps:
            return
        metrics = ot_evaluator.evaluate(T, args.ot_eval_samples)
        log_ot_metrics(out_dir, step + 1, metrics)
        if metrics["D_target"] < best_d_target:
            best_d_target = metrics["D_target"]
            torch.save({"step": step + 1, "metrics": metrics,
                        "T": T.state_dict(), "f": f.state_dict()},
                       os.path.join(out_dir, "best_D_target.pt"))

    set_seed(args.seed)
    T = TransportUNet(base_channels=args.base_channels).to(device)
    f = PotentialResNet(base_channels=args.base_channels).to(device)
    warmup_spectral_norm(f, args.img_size, device=device)

    common = dict(
        n_steps=args.n_steps, lr_T=lr_T, lr_f=lr_f,
        batch_size_x=args.batch_size, batch_size_y=args.batch_size,
        lr_schedule=lr_schedule, lr_schedule_every=1,
        log_every=args.log_every, callback=callback,
        log_dir=out_dir, grad_clip=args.grad_clip, optimizer_kwargs=ADAM_KWARGS,
    )
    sample_mu = lambda batch_size: source_sampler.sample()
    sample_nu = lambda batch_size: target_sampler.sample()
    if args.method == "gdmax":
        history = train_gdmax(T, f, sample_mu, sample_nu, F.mse_loss, K=args.K,
                              resample_each_step=not args.gdmax_reuse_batch, **common)
    else:
        history = train_extragradient(T, f, sample_mu, sample_nu, F.mse_loss,
                                      use_adam=not args.eg_plain_gd, **common)

    torch.save({"T": T.state_dict(), "f": f.state_dict()},
               os.path.join(out_dir, "final.pt"))
    log_ot_metrics(out_dir, args.n_steps,
                   ot_evaluator.evaluate(T, args.final_ot_eval_samples))
    plot_loss(history["loss"], save_path=os.path.join(out_dir, "loss.png"))
    plot_grad_norm(history["grad_T"], history["grad_f"],
                   save_path=os.path.join(out_dir, "grad_norm.png"))
    show_image_samples(T, x_fixed, y_fixed, method_name=args.method,
                       save_path=os.path.join(out_dir, "final_samples.png"))


if __name__ == "__main__":
    main()
