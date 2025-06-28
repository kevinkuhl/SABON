import argparse
import os
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .sabon import SABON


def load_config(yaml_path: str = "config.yaml") -> SimpleNamespace:
    with open(yaml_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if "base_path" in cfg_dict:
        bp = cfg_dict["base_path"]
        cfg_dict.setdefault("grid_path", os.path.join(bp, "grid.npy"))
        cfg_dict.setdefault("x_path", os.path.join(bp, "xdata.npy"))
        cfg_dict.setdefault("y_path", os.path.join(bp, "ydata.npy"))

    return SimpleNamespace(**cfg_dict)


def basis_rank(mats: np.ndarray, rtol: float = 1e-12) -> int:
    mats = np.asarray(mats)
    n_funcs = mats.shape[0] if mats.ndim == 3 else len(mats)
    return np.linalg.matrix_rank(mats.reshape(n_funcs, -1), tol=rtol)


class LpLoss:
    """Relative Lp loss."""

    def __init__(self, p: int = 2, size_average: bool = True, reduction: bool = True):
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.device != y.device:
            y = y.to(x.device)
        num_examples = x.size(0)
        diff = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, dim=1
        )
        ref = torch.norm(y.reshape(num_examples, -1), self.p, dim=1)
        rel_err = diff / (ref + 1e-12)
        if self.reduction:
            return rel_err.mean() if self.size_average else rel_err.sum()
        return rel_err

    __call__ = rel


def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def linear_augmentation(
    n_train: int,
    linear_augment_percentage: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = 1,
):
    rng = np.random.RandomState(random_seed)
    n_augment = int(n_train * linear_augment_percentage)

    alphas = rng.uniform(0, 5, size=n_augment - 1)
    betas = rng.uniform(0, 5, size=n_augment - 1)

    x_aug, y_aug = [], []
    for alpha, beta in zip(alphas, betas):
        i, j = rng.choice(n_train, 2, replace=False)
        x_aug.append(beta * x_train[i] + alpha * x_train[j])
        y_aug.append(beta * y_train[i] + alpha * y_train[j])

    x_train_aug = np.vstack([x_train, np.asarray(x_aug)])
    y_train_aug = np.vstack([y_train, np.asarray(y_aug)])

    print(f"Training data augmented - x:{x_train_aug.shape}, y:{y_train_aug.shape}")
    return x_train_aug, y_train_aug


class Logger:
    def __init__(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        self._file = open(os.path.join(path, f"{filename}.txt"), "w")

    def log(self, message: str):
        self._file.write(message + "\n")
        self._file.flush()
        print(message)


def _str_to_typed(val: str) -> Any:
    low = val.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        if "," in val:
            return [_str_to_typed(v.strip()) for v in val.split(",")]
        return val


def parse_cli() -> Tuple[str, str, Dict[str, Any]]:
    parser = argparse.ArgumentParser(
        description="CLI for SABON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="YAML config file")
    parser.add_argument(
        "--save_dir", default="./checkpoints", help="Root save directory"
    )

    args, unknown = parser.parse_known_args()

    overrides: Dict[str, Any] = {}
    for token in unknown:
        if "=" not in token:
            parser.error(f"unexpected token '{token}'. Use key=value for overrides")
        key, val = token.split("=", 1)
        overrides[key] = _str_to_typed(val)

    return args.config, args.save_dir, overrides


def load_model(checkpoint_dir: str, model_file: str = "best.pth"):
    cfg_path = os.path.join(checkpoint_dir, "config.yaml")
    cfg = load_config(cfg_path)

    grid = np.load(cfg.grid_path)
    grid_flat = grid.reshape(-1, grid.shape[-1])

    t_in = torch.tensor(grid_flat, dtype=torch.float32, device=cfg.device)

    xdata = torch.tensor(np.load(cfg.x_path), dtype=torch.float32)
    ydata = torch.tensor(np.load(cfg.y_path), dtype=torch.float32)

    act_encoder = get_activation(cfg.activation_encoder)
    act_g = get_activation(cfg.activation_g)

    model = SABON(
        d=grid_flat.shape[-1],
        grid_in=grid_flat,
        nbasis=cfg.nbasis,
        encoder_hidden=cfg.encoder_hidden,
        g_hidden=cfg.g_hidden,
        activation_encoder=act_encoder,
        activation_g=act_g,
        device=cfg.device,
    ).to(cfg.device)

    ckpt = torch.load(os.path.join(checkpoint_dir, model_file), map_location=cfg.device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    return model, t_in, grid, xdata, ydata, cfg
