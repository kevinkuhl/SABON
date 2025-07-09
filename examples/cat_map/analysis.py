import argparse
import datetime
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from matplotlib.patches import Circle
from scipy.linalg import subspace_angles

sys.path.append("../..")
import random
import warnings

from galerkin import eigs, galerkin_phiT
from sabon.utils import load_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def savefig(fig, path):
    fig.savefig(path + ".png", dpi=1000, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def l2_l1_normalise(arr: np.ndarray) -> np.ndarray:
    a = arr / np.linalg.norm(arr, 2)
    s = a.sum()
    if s != 0:
        a = a / s
    return a


def plot_eigenvalues(
    result,
    ax=None,
    *,
    fig_size=(8, 6),
    marker="o",
    lead_kw=None,
    unit_kw=None,
    inner_kw=None,
    inflate=0.05,
    label_fs=16,
    tick_fs=10,
    title_fs=18,
    legend_fs=None,
    figfile=None,
    dpi=1000,
    **scatter_kw,
):
    lam = np.asarray(result["eigenvalues"])
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    idx = np.argsort(-np.abs(lam))
    i1, i2 = idx[0], idx[1]
    bulk_mask = np.ones_like(lam, dtype=bool)
    bulk_mask[i1] = False
    ax.scatter(lam[bulk_mask].real, lam[bulk_mask].imag, marker=marker, **scatter_kw)

    lead = dict(color="red", s=64, zorder=5, label="Leading eigenvalue")
    if lead_kw:
        lead.update(lead_kw)
    ax.scatter(lam[i1].real, lam[i1].imag, **lead)

    unit = dict(
        color="green", linestyle="--", linewidth=1.2, fill=False, label="Unit circle"
    )
    if unit_kw:
        unit.update(unit_kw)
    ax.add_patch(Circle((0, 0), 1.0, **unit))

    inner = dict(
        color="grey", linestyle=":", linewidth=1, fill=False, label="|λ₂| circle"
    )
    if inner_kw:
        inner.update(inner_kw)
    ax.add_patch(Circle((0, 0), np.abs(lam[i2]) * (1 + inflate), **inner))

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel(r"Re($\lambda$)", fontsize=label_fs)
    ax.set_ylabel(r"Im($\lambda$)", fontsize=label_fs)
    ax.set_title("Eigenvalues in the complex plane", fontsize=title_fs)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":", lw=0.4)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(axis="both", labelsize=tick_fs)
    if legend_fs is None:
        legend_fs = label_fs
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=legend_fs)
    fig.tight_layout()
    if figfile:
        fig.savefig(figfile, dpi=dpi, bbox_inches="tight")
    return fig


def make_1d_basis(n):
    if n % 2:
        raise ValueError("n must be even")
    root2 = math.sqrt(2.0)
    out = [("c", 0, 1.0)]
    for k in range(1, n // 2 + 1):
        out += [("c", k, root2), ("s", k, root2)]
    return out


def make_2d_basis(n):
    b1d = make_1d_basis(n)
    return [(t1, k1, t2, k2, s1 * s2) for t1, k1, s1 in b1d for t2, k2, s2 in b1d]


def eval_basis_2d(basis2d, X):
    vals = np.empty((len(basis2d), X.shape[0]))
    tp = 2.0 * math.pi
    x1, x2 = X[:, 0], X[:, 1]
    for i, (t1, k1, t2, k2, s) in enumerate(basis2d):
        p1 = tp * k1 * x1
        p2 = tp * k2 * x2
        f1 = np.cos(p1) if t1 == "c" else np.sin(p1)
        f2 = np.cos(p2) if t2 == "c" else np.sin(p2)
        vals[i] = s * f1 * f2
    return vals


def build_basis_arrays(n, grid_factor=4, grid_size=None, delta=0.01):
    basis2d = make_2d_basis(n)
    if grid_size is None:
        m = grid_factor * n
    else:
        m = grid_size
    pts = (np.arange(m) + 0.5) / m
    xx, yy = np.meshgrid(pts, pts, indexing="ij")
    X = np.column_stack((xx.ravel(), yy.ravel()))
    Ny = Nx = m
    Ei = eval_basis_2d(basis2d, X)
    Ej = eval_basis_2d(basis2d, cat_map(X, delta))
    B = [np.rot90(Ei[i].reshape(Ny, Nx, order="F"), 2) for i in range(Ei.shape[0])]
    BT = [np.rot90(Ej[i].reshape(Ny, Nx, order="F"), 2) for i in range(Ej.shape[0])]
    W = np.full((Ny, Nx), 1.0 / (m * m))
    return B, BT, W, basis2d, X, m


def cat_map(x, delta=0.01):
    y0 = 2.0 * x[..., 0] + x[..., 1] + 2.0 * delta * np.cos(2.0 * math.pi * x[..., 0])
    y1 = x[..., 0] + x[..., 1] + delta * np.sin(4.0 * math.pi * x[..., 1] + 1.0)
    return np.mod(np.stack((y0, y1), axis=-1), 1.0)


def principal_angles(basis1, basis2, descending=True):
    A = np.stack([v.ravel() for v in basis1], axis=1).astype(np.float64)
    B = np.stack([v.ravel() for v in basis2], axis=1).astype(np.float64)
    theta = subspace_angles(A, B)

    if descending:
        theta = theta[::-1]

    return theta


def run_angle(model, t_in, device, out_dir):
    grid_size = 100
    phi_raw = model.Encoder(t_in.reshape(-1, 4)).detach().cpu().numpy().T
    phi = [p.reshape(grid_size, grid_size) for p in phi_raw]
    Lphi = [
        model(
            torch.tensor(p).unsqueeze(0).to(device),
            torch.tensor(p).unsqueeze(0).to(device),
        )[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(grid_size, grid_size)
        for p in phi
    ]

    n = 14
    B, BT, _, _, _, _ = build_basis_arrays(n, grid_size=100)
    B = [b.astype(np.float64) for b in B]
    BT = [b.astype(np.float64) for b in BT]

    theta_fourier = principal_angles(B, BT)
    theta_ml = principal_angles(phi, Lphi)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        range(len(theta_fourier)),
        np.sin(theta_fourier)[np.argsort(theta_fourier)],
        marker="x",
        label="Fourier basis",
    )
    ax.scatter(
        range(len(theta_ml)),
        np.sin(theta_ml)[np.argsort(theta_ml)],
        marker="x",
        c="red",
        label="SABON basis",
    )
    ax.set_xlabel("Principal-angle index k", fontsize=16)
    ax.set_ylabel(r"Projection distance $\sin\theta_k$", fontsize=16)
    ax.set_title("Per-angle projection distance", fontsize=18)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=16)
    savefig(fig, os.path.join(out_dir, "anosov_angle"))

    def distance_metrics(th):
        sin_th = np.sin(th)
        return dict(
            projection=sin_th.max(),
            chordal=np.linalg.norm(sin_th),
            geodesic=np.linalg.norm(th),
        )

    print("--- Principal-subspace distances (radians) ---")
    dist_four = distance_metrics(theta_fourier)
    dist_ml = distance_metrics(theta_ml)
    for k in dist_four:
        print(f"{k:10s}: Fourier = {dist_four[k]:.6f} | SABON = {dist_ml[k]:.6f}")


def plot_basis_and_gram(model, t_in, out_dir):
    EXTENT = (0, 1, 0, 1)
    TICKS = [0, 0.5, 1]
    TICK_LABELS = ["0", "0.5", "1"]

    def style_axes(ax, show_frame=True):
        ax.set_xticks(TICKS, TICK_LABELS, fontsize=11)
        ax.set_yticks(TICKS, TICK_LABELS, fontsize=11)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_aspect("equal")
        if not show_frame:
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    B_flat = model.Encoder(t_in.reshape(-1, 4)).detach().cpu().numpy().T
    n, HW = B_flat.shape
    norms = np.linalg.norm(B_flat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    B_flat_norm = B_flat / norms
    gram = B_flat_norm @ B_flat_norm.T

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(gram, cmap="bwr", origin="lower", vmin=-1, vmax=1)
    ax.set_xlabel("Basis index", fontsize=20)
    ax.set_ylabel("Basis index", fontsize=20)
    cbar = fig.colorbar(im, ax=ax, location="right", shrink=0.8, pad=0.03)
    cbar.set_label(r"$M_{kj}$", fontsize=20)
    savefig(fig, os.path.join(out_dir, "anosov_gram_matrix"))

    H = W = 100
    sel = [i for i in [0, 50, 100, 150, 200] if i < n]
    patches = B_flat[sel].reshape(len(sel), H, W)
    vmax = np.abs(
        patches / np.linalg.norm(patches.reshape(len(sel), -1), axis=1)[:, None, None]
    ).max()

    fig, axes = plt.subplots(
        1, len(sel), figsize=(3 * len(sel), 3), constrained_layout=True
    )
    for i, (ax, p) in enumerate(zip(axes, patches)):
        norm_p = p / np.linalg.norm(p)
        im = ax.imshow(
            norm_p, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax, extent=EXTENT
        )
        style_axes(ax, show_frame=False)
        ax.set_title(rf"$\phi_{{{sel[i]}}}$", fontsize=22)

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), location="right", shrink=0.8, pad=0.02
    )
    cbar.set_label("Normalised basis value", rotation=270, labelpad=14, fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    savefig(fig, os.path.join(out_dir, "anosov_basis"))


def plot_single_prediction(model, x_data, y_data, out_dir):
    EXTENT = (0, 1, 0, 1)
    TICKS = [0, 0.5, 1]
    TICK_LABELS = ["0", "0.5", "1"]

    label_fs = 12
    tick_fs = 11
    title_fs = 14

    def style(ax):
        ax.set_xticks(TICKS, TICK_LABELS, fontsize=tick_fs)
        ax.set_yticks(TICKS, TICK_LABELS, fontsize=tick_fs)
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.tick_params(direction="in")
        ax.set_aspect("equal")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    device = next(model.parameters()).device
    model.eval()

    test_start_idx = len(x_data) - 500
    test_errors = []

    with torch.no_grad():
        for i in range(test_start_idx, len(x_data)):
            pred = model(
                x_data[i].unsqueeze(0).to(device), x_data[i].unsqueeze(0).to(device)
            )[0]
            pred_np = pred.cpu().squeeze().numpy()
            gt_np = y_data[i].cpu().squeeze().numpy()

            l2_error = np.linalg.norm(pred_np - gt_np.ravel())
            l2_norm_gt = np.linalg.norm(gt_np)
            relative_err = l2_error / l2_norm_gt if l2_norm_gt > 0 else 0
            test_errors.append(relative_err)

    mean_rel_l2_error = np.mean(test_errors)

    with torch.no_grad():
        pred = model(
            x_data[-1].unsqueeze(0).to(device), x_data[-1].unsqueeze(0).to(device)
        )[0]

    pred_np = pred.cpu().squeeze().numpy().reshape(100, 100)
    gt_np = y_data[-1].cpu().squeeze().numpy().reshape(100, 100)
    inp_np = x_data[-1].cpu().squeeze().numpy().reshape(100, 100)

    clim = [min(gt_np.min(), pred_np.min()), max(gt_np.max(), pred_np.max())]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].imshow(inp_np, cmap="bwr", origin="lower", extent=EXTENT)
    axes[0].set_title(r"Input observable $g$", fontsize=title_fs)
    style(axes[0])

    axes[1].imshow(
        gt_np, cmap="bwr", origin="lower", vmin=clim[0], vmax=clim[1], extent=EXTENT
    )
    axes[1].set_title("Ground truth", fontsize=title_fs)
    style(axes[1])

    axes[2].imshow(
        pred_np, cmap="bwr", origin="lower", vmin=clim[0], vmax=clim[1], extent=EXTENT
    )
    axes[2].set_title("Predicted", fontsize=title_fs)
    style(axes[2])

    mappable = axes[1].images[0]
    cbar = fig.colorbar(
        mappable, ax=axes.ravel().tolist(), shrink=0.85, location="right"
    )
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    savefig(fig, os.path.join(out_dir, "anosov_input_output"))
    return mean_rel_l2_error


def run_spectrum(model, t_in, device, mu_path, out_dir):
    grid_size = 100
    phi_raw = model.Encoder(t_in.reshape(-1, 4)).detach().cpu().numpy().T
    phi = [p.reshape(grid_size, grid_size) for p in phi_raw]
    Lphi = [
        model(
            torch.tensor(p).unsqueeze(0).to(device),
            torch.tensor(p).unsqueeze(0).to(device),
        )[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(grid_size, grid_size)
        for p in phi
    ]
    result_ml = eigs(
        model,
        t_in.reshape(-1, 4),
        normalize_eigenvectors=True,
        return_left_vectors=True,
    )
    fig_ev = plot_eigenvalues(result_ml)
    savefig(fig_ev, os.path.join(out_dir, "anosov_eigenvalues"))

    f_hat_ml = result_ml["eigenvectors"][:, 0]
    eig_ml = np.zeros((grid_size, grid_size), dtype=complex)
    for i in range(grid_size):
        for j in range(grid_size):
            eig_ml[i, j] = np.dot(f_hat_ml, [p[i, j] for p in phi])
    eig_ml = eig_ml.real

    n_four = 14
    B, BT, W, basis2d, X, m = build_basis_arrays(n_four, grid_size=100)
    B = [b.astype(np.float64) for b in B]
    BT = [b.astype(np.float64) for b in BT]
    W = W.astype(np.float64)
    result_four = galerkin_phiT(
        B,
        BT,
        inner_product_weight=W,
        normalize_eigenvectors=True,
        return_left_vectors=False,
    )
    f_hat_four = result_four["eigenvectors"][:, 0]
    eig_four = np.zeros((m, m), dtype=complex)
    for i in range(m):
        for j in range(m):
            eig_four[i, j] = np.dot(f_hat_four, [b[i, j] for b in B])
    eig_four = eig_four.real

    mu = np.load(mu_path)

    gt_norm = l2_l1_normalise(mu.copy())
    ml_norm = l2_l1_normalise(eig_ml.copy())
    four_norm = l2_l1_normalise(eig_four.copy())

    def projection_metrics(basis_raw, X, w, mu_vec):
        norms = np.linalg.norm(basis_raw, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Bn = basis_raw / norms
        M = (Bn * w) @ Bn.T
        b = (Bn * w) @ mu_vec
        coeff = np.linalg.solve(M, b)
        proj = coeff @ Bn
        rel_err = math.sqrt(((mu_vec - proj) ** 2 * w).sum()) / math.sqrt(
            (mu_vec**2 * w).sum()
        )
        p = 1
        mu_proj = proj.reshape(mu.shape, order="C")
        mu_norm = l2_l1_normalise(mu.copy())
        mu_proj_norm = l2_l1_normalise(mu_proj.copy())
        xs, ys = np.meshgrid(
            np.arange(mu.shape[0]) / mu.shape[0],
            np.arange(mu.shape[1]) / mu.shape[1],
            indexing="ij",
        )
        coords = np.stack([xs.ravel(), ys.ravel()], axis=1)
        dx = np.abs(coords[:, None, 0] - coords[None, :, 0])
        dx = np.minimum(dx, 1 - dx)
        dy = np.abs(coords[:, None, 1] - coords[None, :, 1])
        dy = np.minimum(dy, 1 - dy)
        C = (dx**2 + dy**2) ** 0.5**p
        Wp = ot.emd2(mu_norm.ravel(), mu_proj_norm.ravel(), C) ** (1 / p)
        return rel_err, Wp

    # SABON metrics
    mu_vec = mu.ravel(order="C")
    w_uniform = np.full_like(mu_vec, 1 / mu_vec.size)
    rel_ml, W1_ml = projection_metrics(
        phi_raw, X=np.empty((0)), w=w_uniform, mu_vec=mu_vec
    )

    # Fourier metrics
    phi_four_raw = eval_basis_2d(basis2d, X)
    rel_four, W1_four = projection_metrics(
        phi_four_raw, X=X, w=np.full(X.shape[0], 1 / X.shape[0]), mu_vec=mu_vec
    )

    print("-------------------------------------------------------------")
    print(f"--> SABON basis   : L2 rel error = {rel_ml:9.3e}   | W1 = {W1_ml:9.3e}")
    print(f"--> Fourier basis : L2 rel error = {rel_four:9.3e} | W1 = {W1_four:9.3e}")
    print("-------------------------------------------------------------")

    # SRB plot
    vmin = min(gt_norm.min(), ml_norm.min(), four_norm.min())
    vmax = max(gt_norm.max(), ml_norm.max(), four_norm.max())
    extent = (0, 1, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ims = []
    ims.append(
        axes[0].imshow(
            gt_norm, vmin=vmin, vmax=vmax, origin="lower", cmap="Blues", extent=extent
        )
    )
    axes[0].set_title("Ground-truth SRB measure")
    ims.append(
        axes[1].imshow(
            np.rot90(ml_norm, 2),
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap="Blues",
            extent=extent,
        )
    )
    axes[1].set_title("Reconstructed measure (SABON)")
    ims.append(
        axes[2].imshow(
            four_norm, vmin=vmin, vmax=vmax, origin="lower", cmap="Blues", extent=extent
        )
    )
    axes[2].set_title("Reconstructed measure (Fourier)")

    for ax in axes:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_aspect("equal")

    fig.colorbar(ims[0], ax=axes.ravel().tolist(), location="right", shrink=0.8)
    savefig(fig, os.path.join(out_dir, "anosov_srb"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mu_path", default="mu_measure_unit.npy")
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.output_dir, ts)
    for sub in ("angle", "plots", "spectrum"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model, t_in, _, x_data, y_data, config = load_model(
        args.checkpoint_dir, model_file="best.pth"
    )
    model.mp_dtype = torch.float32

    device = config.device
    print("Loaded model from", args.checkpoint_dir)
    print("Model config:", config)
    print("Computing angle statistics...")
    run_angle(model, t_in, device, os.path.join(root, "angle"))
    print("Computing basis and gram matrix visualisations...")
    plot_basis_and_gram(model, t_in, os.path.join(root, "plots"))
    print("Computing single prediction plot and test metrics...")
    mean_rel_l2_error = plot_single_prediction(
        model, x_data, y_data, os.path.join(root, "plots")
    )
    print("Computing spectrum, SRB and projection metrics...")
    run_spectrum(model, t_in, device, args.mu_path, os.path.join(root, "spectrum"))

    print("-------------------------------------------------------------")
    print(f"--> Test set mean relative L2 error: {mean_rel_l2_error:.6e}")
    print("-------------------------------------------------------------")


if __name__ == "__main__":
    main()
