# sabon.py

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
    w = torch.full((n, n), h * h, dtype=torch.float32, device=device)
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
        h = trap_step if trap_step is not None else (1.0 / n_pts)
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

    def get_bases(self) -> torch.Tensor:
        """
        Get current basis functions.

        Returns:
            bases: (n_basis, L) tensor of basis functions evaluated on grid
        """
        return self.Encoder(self.t_in).T.contiguous()

    def project(self, x_flat: torch.Tensor, bases: torch.Tensor = None) -> torch.Tensor:
        """Project functions onto the learned basis."""
        if bases is None:
            bases = self.get_bases()
        bases_w = bases * self.trap_w_flat
        return _project_flat(x_flat, bases_w)

    def reconstruct(
        self, coeffs: torch.Tensor, bases: torch.Tensor = None
    ) -> torch.Tensor:
        """Reconstruct functions from coefficients."""
        if bases is None:
            bases = self.get_bases()
        return _reconstruct_flat(coeffs, bases)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Full forward pass.

        Returns:
            out_flat: (batch, L) predicted L(x)
            aec_in_flat: (batch, L) reconstructed x
            aec_out_flat: (batch, L) reconstructed y
            bases: (n_basis, L) basis functions
        """
        dev = self.t_in.device
        x = x.to(dev)
        y = y.to(dev)

        use_amp = self.mp_dtype != torch.float32 and dev.type == "cuda"

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=self.mp_dtype):
            B_in, J1, J2 = x.shape
            B_out = y.size(0)
            L = J1 * J2

            # Get basis
            bases_f32 = self.get_bases()
            bases_mp = bases_f32.to(self.mp_dtype)

            # Project and reconstruct
            x_flat_mp = x.view(B_in, L).to(self.mp_dtype)
            s_in = self.project(x_flat_mp, bases_mp)
            s_out_mp = self.G(s_in.float()).to(self.mp_dtype)

            out_flat_mp = self.reconstruct(s_out_mp, bases_mp)
            aec_in_flat_mp = self.reconstruct(s_in, bases_mp)

            y_flat_mp = y.view(B_out, L).to(self.mp_dtype)
            s_y = self.project(y_flat_mp, bases_mp)
            aec_out_flat_mp = self.reconstruct(s_y, bases_mp)

        return (
            out_flat_mp.float(),
            aec_in_flat_mp.float(),
            aec_out_flat_mp.float(),
            bases_f32.float(),
        )

    def predict_k_steps(
        self, x_flat: torch.Tensor, k: int, bases: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Apply the learned operator k times.

        Computes: reconstruct(G^k(project(x)))

        Args:
            x_flat: (batch, L) input functions
            k: number of steps
            bases: (n_basis, L) basis functions. If None, computed from Encoder.

        Returns:
            out_flat: (batch, L) predicted L^k(x)
        """
        if bases is None:
            bases = self.get_bases()

        # Project
        coeffs = self.project(x_flat, bases)

        # Apply G k times
        for _ in range(k):
            coeffs = self.G(coeffs)

        # Reconstruct
        return self.reconstruct(coeffs, bases)

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
