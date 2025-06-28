import math
import os
from typing import List, Tuple

import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(
        self,
        hidden: List[int] = (64, 64),
        dim_in: int = -1,
        dim_out: int = -1,
        activation=None,
        bias: bool = False,
    ):
        super().__init__()
        self.act = activation or nn.ReLU()
        sizes = [dim_in, *hidden, dim_out]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class NeuralBasis(nn.Module):
    def __init__(
        self,
        dim_in: int = 1,
        hidden: List[int] = (4, 4, 4),
        nbasis: int = 4,
        activation=None,
        bias: bool = False,
    ):
        super().__init__()
        self.act = activation or nn.Tanh()
        sizes = [dim_in, *hidden, nbasis]
        self.layers = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1], bias=bias) for i in range(len(sizes) - 1)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            t = self.act(layer(t))
        return self.layers[-1](t)


def _select_mp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return (
        torch.bfloat16
        if major >= 8
        else (torch.float16 if major >= 7 else torch.float32)
    )


def _make_trap_weights(n: int, h: float, device, dtype) -> torch.Tensor:
    w = torch.ones((n, n), dtype=torch.float32, device=device)
    w[[0, -1], :] *= 0.5
    w[:, [0, -1]] *= 0.5
    w[0, 0] = w[0, -1] = w[-1, 0] = w[-1, -1] = 0.25
    w *= h * h
    return w.to(dtype).flatten()


def _project_flat(x: torch.Tensor, bases_w: torch.Tensor) -> torch.Tensor:
    return x @ bases_w.T


def _reconstruct_flat(coeff: torch.Tensor, bases: torch.Tensor) -> torch.Tensor:
    return coeff @ bases


class SABON(nn.Module):
    def __init__(
        self,
        d: int,  # dimension of the input/output space
        grid_in: torch.Tensor,
        nbasis: int = 9,
        encoder_hidden: Tuple[int, ...] = (64, 64, 64),
        g_hidden: Tuple[int, ...] = (64, 64, 64),
        activation_encoder=None,
        activation_g=None,
        trap_step: float = None,
        device: str = None,
    ):
        super().__init__()
        if d not in (2, 4):
            raise ValueError("d must be 2 or 4")

        self._device = torch.device(device) if device else torch.device("cpu")

        self.nbasis = nbasis
        self.mp_dtype = _select_mp_dtype()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.register_buffer(
            "t_in", torch.as_tensor(grid_in, dtype=torch.float32).view(-1, d)
        )

        n_pts = int(math.sqrt(grid_in.shape[0]))
        h = trap_step if trap_step is not None else (2.0 * math.pi / n_pts)
        self.register_buffer(
            "trap_w_flat", _make_trap_weights(n_pts, h, self._device, self.mp_dtype)
        )

        self.Encoder = NeuralBasis(
            d,
            hidden=list(encoder_hidden),
            nbasis=nbasis,
            activation=activation_encoder,
            bias=True,
        )
        self.G = FNN(
            hidden=list(g_hidden),
            dim_in=nbasis,
            dim_out=nbasis,
            activation=activation_g,
            bias=False,
        )

        self.to(self._device)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """out_flat, aec_in_flat, aec_out_flat, bases_flat"""
        dev = self.t_in.device
        x = x.to(dev)
        y = y.to(dev)

        use_amp = self.mp_dtype != torch.float32 and dev.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.mp_dtype):
            B_in, J1, J2 = x.shape
            B_out = y.size(0)
            L = J1 * J2

            bases_f32 = self.Encoder(self.t_in).T.contiguous()
            bases_mp = bases_f32.to(self.mp_dtype)
            bases_w_flat = bases_mp * self.trap_w_flat

            x_flat_mp = x.view(B_in, L).to(self.mp_dtype)
            s_in = _project_flat(x_flat_mp, bases_w_flat)
            s_out_mp = self.G(s_in.float()).to(self.mp_dtype)

            out_flat_mp = _reconstruct_flat(s_out_mp, bases_mp)
            aec_in_flat_mp = _reconstruct_flat(s_in, bases_mp)
            y_flat_mp = y.view(B_out, L).to(self.mp_dtype)
            s_y = _project_flat(y_flat_mp, bases_w_flat)
            aec_out_flat_mp = _reconstruct_flat(s_y, bases_mp)

        return (
            out_flat_mp.float(),
            aec_in_flat_mp.float(),
            aec_out_flat_mp.float(),
            bases_f32.float(),
        )

    def load_ckpt(
        self,
        path: str,
        optimizer: torch.optim.Optimizer = None,
        map_to_cpu: bool = False,
    ) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        ckpt = torch.load(
            path, map_location=torch.device("cpu") if map_to_cpu else None
        )
        model_sd: dict[str, torch.Tensor] = ckpt.get("model_state", ckpt)

        if optimizer and (opt_sd := ckpt.get("optimizer_state")):
            try:
                optimizer.load_state_dict(opt_sd)
                opt_msg = ", optimizer successfully loaded."
            except Exception:
                opt_msg = ", optimizer state dict incompatible."
        else:
            opt_msg = "."

        missing, unexpected = self.load_state_dict(model_sd, strict=False)
        n_loaded = len(model_sd) - len(missing)
        print(f"--> SABON Loaded {n_loaded}/{len(self.state_dict())} tensors{opt_msg}")
        if missing:
            print("missing:", len(missing))
        if unexpected:
            print("unexpected:", len(unexpected))
