"""
Usage
-----
python train.py                              # defaults: config.yaml, ./checkpoints
python train.py --config run.yaml            # custom YAML
python train.py --config run.yaml --save_dir ./my_checkpoints
"""

import os
import sys
import time
import warnings
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append("../..")
from sabon.sabon import SABON
from sabon.utils import (
    Logger,
    LpLoss,
    get_activation,
    linear_augmentation,
    load_config,
    parse_cli,
)


def _save_yaml(ns: SimpleNamespace, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(vars(ns), f)


def _invariance_loss(
    basis: torch.Tensor, trap_w: torch.Tensor, A_hat: torch.Tensor
) -> torch.Tensor:
    """Relative Frobenius error enforcing invariance."""
    Lt = (A_hat @ basis.T).T
    coeff = Lt @ (basis * trap_w).T
    recon = coeff @ basis
    err = recon - Lt
    return torch.norm(err) / (torch.norm(Lt).detach() + 1e-12)


def _print_main_params(cfg: SimpleNamespace, logger: Logger) -> None:
    keys = [
        "model_name",
        "nbasis",
        "batch_size",
        "learning_rate",
        "epochs",
        "scheduler_step_size",
        "gamma",
        "lambda_op",
        "lambda_in",
        "lambda_out",
        "lambda_sparse",
        "lambda_inv",
        "device",
    ]
    logger.log("--> PARAMETERS")
    for k in keys:
        if hasattr(cfg, k):
            logger.log(f"{k}:{getattr(cfg, k)}")


def train(
    *, cfg_path: str = "config.yaml", save_dir: str = "./checkpoints", **override: Any
):
    cfg = load_config(cfg_path)
    for k, v in override.items():
        setattr(cfg, k, v)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"{cfg.model_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(run_dir, "train")

    _print_main_params(cfg, logger)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.random_seed)

    x = np.load(cfg.x_path)
    y = np.load(cfg.y_path)
    grid = np.load(cfg.grid_path)
    grid = grid[:: cfg.sub, :: cfg.sub, :]
    grid_flat = grid.reshape(-1, grid.shape[-1])

    ntr, nval, nte = cfg.ntrain, cfg.nvalid, cfg.ntest
    x_tr, x_va, x_te = np.split(x, [ntr, ntr + nval])
    y_tr, y_va, y_te = np.split(y, [ntr, ntr + nval])

    if cfg.linear_augment:
        x_tr, y_tr = linear_augmentation(
            ntr, cfg.linear_augment_percentage, x_tr, y_tr, cfg.random_seed
        )
        ntr = x_tr.shape[0]

    x_tr = torch.from_numpy(x_tr).float()
    x_va = torch.from_numpy(x_va).float()
    x_te = torch.from_numpy(x_te).float()
    y_tr = torch.from_numpy(y_tr).float()
    y_va = torch.from_numpy(y_va).float()
    y_te = torch.from_numpy(y_te).float()

    J1, J2 = y_tr.shape[1:3]
    Ldim = J1 * J2

    # pre‑compute operator surrogate
    with torch.no_grad():
        F = x_tr.reshape(ntr, Ldim).T
        LF = y_tr.reshape(ntr, Ldim).T
        A_hat = (LF @ torch.linalg.pinv(F)).to(device)

    tr_loader = DataLoader(
        TensorDataset(x_tr, y_tr), batch_size=cfg.batch_size, shuffle=True
    )
    va_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=cfg.batch_size)
    te_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=cfg.batch_size)

    act_encoder = get_activation(cfg.activation_encoder)
    act_g = get_activation(cfg.activation_g)

    model = SABON(
        d=grid.shape[-1],
        grid_in=torch.from_numpy(grid_flat),
        nbasis=cfg.nbasis,
        encoder_hidden=cfg.encoder_hidden,
        g_hidden=cfg.g_hidden,
        activation_encoder=act_encoder,
        activation_g=act_g,
        device=device,
    ).to(device)

    logger.log("--> model instantiated")

    mse_loss = nn.MSELoss()
    rel_loss = LpLoss(size_average=False)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(
        optim, step_size=cfg.scheduler_step_size, gamma=cfg.gamma
    )

    ckpt_path = os.path.join(run_dir, "best.pth")
    best_val = float("inf")

    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        tr_tot = tr_op = tr_aec_in = tr_aec_out = tr_inv = 0.0

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            out, a_in, a_out, bases = model(xb, yb)

            inv_loss = _invariance_loss(bases, model.trap_w_flat, A_hat)
            op_loss = rel_loss(out, yb.view(-1, Ldim))
            a_in_loss = rel_loss(a_in, xb)
            a_out_loss = rel_loss(a_out, yb)
            sp_loss = torch.norm(bases, 1, 1).mean()

            loss = (
                cfg.lambda_op * op_loss
                + cfg.lambda_in * a_in_loss
                + cfg.lambda_out * a_out_loss
                + cfg.lambda_sparse * sp_loss
                + cfg.lambda_inv * inv_loss
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            optim.step()

            tr_tot += loss.item()
            tr_op += op_loss.item()
            tr_aec_in += a_in_loss.item()
            tr_aec_out += a_out_loss.item()
            tr_inv += inv_loss.item()

        sched.step()
        tr_tot /= ntr
        tr_op /= ntr
        tr_aec_in /= ntr
        tr_aec_out /= ntr
        tr_inv /= ntr

        model.eval()
        va_tot = va_op = va_aec_in = va_aec_out = va_inv = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                out, a_in, a_out, bases = model(xb, yb)

                inv_loss = _invariance_loss(bases, model.trap_w_flat, A_hat)
                op_loss = rel_loss(out, yb.view(-1, Ldim))
                a_in_loss = rel_loss(a_in, xb)
                a_out_loss = rel_loss(a_out, yb)
                sp_loss = torch.norm(bases, 1, 1).mean()

                loss = (
                    cfg.lambda_op * op_loss
                    + cfg.lambda_in * a_in_loss
                    + cfg.lambda_out * a_out_loss
                    + cfg.lambda_sparse * sp_loss
                    + cfg.lambda_inv * inv_loss
                )
                va_tot += loss.item()
                va_op += op_loss.item()
                va_aec_in += a_in_loss.item()
                va_aec_out += a_out_loss.item()
                va_inv += inv_loss.item()

        va_tot /= nval
        va_op /= nval
        va_aec_in /= nval
        va_aec_out /= nval
        va_inv /= nval
        dt = time.time() - t0

        logger.log(
            f"ep:{ep:04d} | lr:{optim.param_groups[0]['lr']:.7f} | time:{dt:.2f} | "
            f"train_tot:{tr_tot:.7f} | train_op:{tr_op:.7f} | train_aec_in:{tr_aec_in:.7f} | "
            f"train_aec_out:{tr_aec_out:.7f} | train_inv:{tr_inv:.7f} | "
            f"valid_tot:{va_tot:.7f} | valid_op:{va_op:.7f} | valid_aec_in:{va_aec_in:.7f} | "
            f"valid_aec_out:{va_aec_out:.7f} | valid_inv:{va_inv:.7f} | nbasis:{model.nbasis}"
        )

        if va_op < best_val:
            best_val = va_op
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optim.state_dict(),
                },
                ckpt_path,
            )
            logger.log("new_best_model")
            _save_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    model.load_ckpt(ckpt_path)
    model.eval()
    mse_op = 0.0
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            out, _, _, _ = model(xb, yb)
            mse_op += mse_loss(out, yb.view(-1, Ldim)).item()
    mse_op /= nte
    logger.log(f"test_mse_op:{mse_op:.6e}")


if __name__ == "__main__":
    cfg_path, save_dir, overrides = parse_cli()
    train(cfg_path=cfg_path, save_dir=save_dir, **overrides)
