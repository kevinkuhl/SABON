import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from numpy.linalg import qr, svd

sys.path.append("../..")
import warnings

from galerkin import galerkin_Lphi
from sabon.utils import LpLoss, load_model

warnings.filterwarnings("ignore", category=FutureWarning)
ALPHA = 1.0


def make_peaks_positive(basis: np.ndarray) -> np.ndarray:
    basis = basis.copy()
    mask = np.abs(basis).max(axis=1) == -basis.min(axis=1)
    basis[mask] *= -1
    return basis


def apply_operator(model: torch.nn.Module, vec_1d: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    x_t = torch.as_tensor(vec_1d, dtype=torch.float32, device=dev)
    x_t = x_t.unsqueeze(1).unsqueeze(0)
    with torch.no_grad():
        y_t = model(x_t, x_t)[0].squeeze()
    return y_t.cpu().numpy()


def plot_bases_heatmaps(
    basis_list,
    grid,
    titles,
    thr=0.15,
    cmap="bwr",
    basename="basis_heatmaps",
    formats=("png", "pdf"),
    dpi=1000,
):
    def _sort(mat):
        t = np.abs(mat)
        gc = grid[:, 0] + 1j * grid[:, 1]
        ang = np.angle((t * gc[None, :]).sum(1)) % (2 * np.pi)
        nrm = np.linalg.norm(t, axis=1)
        sig = nrm > thr * nrm.max()
        order = np.r_[np.where(sig)[0][np.argsort(ang[sig])], np.where(~sig)[0]]
        return mat[order]

    mats = [_sort(b) for b in basis_list]
    vmax = max(np.abs(m).max() for m in mats)
    vmin = -vmax
    fig, axes = plt.subplots(
        1, len(mats), figsize=(6 * len(mats), 6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    N = mats[0].shape[1]

    for ax, mat, tl in zip(axes, mats, titles):
        im = ax.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
        )
        ax.set_title(tl, fontsize=22)
        ax.set_xlabel(r"$\theta$ (grid points)", fontsize=22)
        if ax is axes[0]:
            ax.set_ylabel("Basis index", fontsize=22)
        ax.set_xticks([0, N // 4, N // 2, 3 * N // 4, N - 1])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
        ax.set_yticks([0, 9, 19, 29])
        ax.set_yticklabels([1, 10, 20, 30])

    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.03)
    cb.set_label(r"$\phi_j$", fontsize=18)

    for ext in formats:
        fn = f"{basename}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        print(f"Saved {fn}")
    plt.close(fig)


def compute_gram_matrices(basis_list):
    return [b @ b.T for b in basis_list]


def plot_gram_heatmaps(
    gram_list,
    titles,
    cmap="bwr",
    basename="gram_matrices",
    formats=("png", "pdf"),
    dpi=1000,
):
    vmax = max(np.abs(g).max() for g in gram_list)
    vmin = -vmax
    fig, axes = plt.subplots(
        1, len(gram_list), figsize=(6 * len(gram_list), 6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, G, tl in zip(axes, gram_list, titles):
        im = ax.imshow(
            G, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="equal"
        )
        ax.set_title(tl, fontsize=22)
        ax.set_xlabel("Basis index", fontsize=22)
        if ax is axes[0]:
            ax.set_ylabel("Basis index", fontsize=22)

    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.9, pad=0.03)
    cb.set_label(r"$G_{kj}$", fontsize=18)

    for ext in formats:
        fn = f"{basename}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        print(f"Saved {fn}")
    plt.close(fig)


def plot_predictions_grid(
    xdata,
    ydata,
    models,
    model_titles=None,
    basename="circle_rotation_input_output",
    dpi=1000,
):
    num = len(models)
    n_cols, n_rows = 2, int(np.ceil(num / 2))
    figsize = (6 * n_cols, 4 * n_rows)
    if model_titles is None:
        model_titles = [f"Model {i + 1}" for i in range(num)]

    sample = xdata[-1]
    N = sample.numel()
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig)

    xticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    xtick_labels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    mean_errors = []

    for idx, model in enumerate(models):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])

        xd = xdata[-1]
        yd = ydata[-1]
        dev = next(model.parameters()).device
        xd_dev = xd.to(dev)

        with torch.no_grad():
            yp = model(xd_dev.unsqueeze(0), xd_dev.unsqueeze(0))[0].squeeze()

        xd_np = xd.cpu().numpy().ravel()
        yd_np = yd.cpu().numpy().ravel()
        yp_np = yp.cpu().numpy().ravel()

        if idx == 0:
            ax.plot(theta, xd_np, color="tab:green", lw=2, label="Input f(θ)")
            ax.plot([np.pi / 2 + 0.35, np.pi / 2 + 1.35], [2.5, 2.5], lw=2, color="k")
            ax.plot([np.pi / 2 + 0.35] * 2, [2.4, 2.6], color="k")
            ax.plot([np.pi / 2 + 1.35] * 2, [2.4, 2.6], color="k")
            ax.text(np.pi / 2 + 0.85, 2.65, r"$\Delta\theta=1$", ha="center")

        ax.plot(theta, yd_np, color="tab:orange", lw=2, label="Ground truth")
        ax.plot(theta, yp_np, color="tab:blue", lw=2, ls="--", label="Prediction")

        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
        ax.grid(True)
        ax.set_title(model_titles[idx], fontsize=15)
        if col == 0:
            ax.set_ylabel(r"$\mathcal{L}(f(\theta))$", fontsize=12)
        if row == n_rows - 1:
            ax.set_xlabel(r"$\theta$ (grid points)", fontsize=12)
        ax.legend(loc="upper left", fontsize=8)

    lp_rel = LpLoss(p=2, size_average=False, reduction=True)
    last = slice(max(len(xdata) - 100, 0), None)
    B = len(xdata[last])

    mean_errors = []

    for mdl, title in zip(models, model_titles):
        mdl.eval()
        dev = next(mdl.parameters()).device

        X_batch = torch.stack([xi.to(dev) for xi in xdata[last]])
        Y_batch = torch.stack([yi.to(dev) for yi in ydata[last]])

        with torch.no_grad():
            Y_pred = mdl(X_batch, X_batch)[0].squeeze(-1)
        Y_true = Y_batch.squeeze(-1)

        rel_sum = lp_rel.rel(Y_pred, Y_true).item()
        mean_err = rel_sum / B
        mean_errors.append(mean_err)

    for ext in ("png", "pdf"):
        fn = f"{basename}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        print(f"Saved {fn}")
    plt.close(fig)

    print("\nMean relative L² error on last-100 functions:")
    for t, e in zip(model_titles, mean_errors):
        print(f"  {t:25s}: {e:.5e}")

    return mean_errors


def plot_eigenvalue_spectrum(
    galerkin_result,
    basename="spectrum_eigvals",
    formats=("png", "pdf"),
    dpi=1000,
):
    label_fs, tick_fs, title_fs, legend_fs = 16, 14, 18, 15
    figsize, margin = (18, 7), 0.05
    n_max = 10

    lam_num = np.asarray(galerkin_result["eigenvalues"])

    n_all = np.arange(-n_max, n_max + 1)
    lam_th = np.exp(1j * n_all)
    lam_lead_th = np.exp(1j * np.array([1, -1]))

    tol_ang = 1e-2
    mask_lead = np.zeros_like(lam_num, dtype=bool)
    for lam in lam_lead_th:
        mask_lead |= np.abs(lam_num - lam) < tol_ang

    lam_num_lead = lam_num[mask_lead]
    lam_num_oth = lam_num[~mask_lead]

    fig, (axC, axPA) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.25]}
    )

    sc_other = axC.scatter(
        lam_num_oth.real,
        lam_num_oth.imag,
        s=75,
        marker="o",
        facecolors="none",
        edgecolors="tab:blue",
        linewidths=1.5,
    )
    sc_lead = axC.scatter(
        lam_num_lead.real,
        lam_num_lead.imag,
        s=75,
        marker="o",
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.5,
    )
    sc_th = axC.scatter(lam_th.real, lam_th.imag, marker="x", s=60, c="tab:green")

    axC.axhline(0, color="k", lw=0.5)
    axC.axvline(0, color="k", lw=0.5)
    axC.set_xlabel(r"Re$(\lambda)$", fontsize=label_fs)
    axC.set_ylabel(r"Im$(\lambda)$", fontsize=label_fs)
    axC.set_title("Eigenvalues in the complex plane", fontsize=title_fs)
    axC.set_xticks([-1, -0.5, 0, 0.5, 1])
    axC.set_yticks([-1, -0.5, 0, 0.5, 1])
    axC.tick_params(axis="both", labelsize=tick_fs)
    axC.set_box_aspect(1)
    axC.set_aspect("equal", adjustable="box")
    low, high = -1 - margin, 1 + margin
    axC.set_xlim(low, high)
    axC.set_ylim(low, high)
    axC.grid(True, ls=":", lw=0.4)

    arg_num, abs_num = np.angle(lam_num), np.abs(lam_num)
    arg_lead, abs_lead = np.angle(lam_num_lead), np.abs(lam_num_lead)

    axPA.scatter(arg_num, abs_num, s=30, c="tab:blue")
    axPA.scatter(arg_lead, abs_lead, s=40, c="tab:red")

    mask_guides = (arg_num < 0) & np.isclose(abs_num, 1.0, atol=0.02)
    for a in arg_num[mask_guides]:
        axPA.axvline(a, color="tab:green", ls="--", lw=0.8)

    xt_new = sorted(set(chain(axPA.get_xticks(), arg_num[mask_guides])))
    axPA.set_xticks(xt_new)

    axPA.set_xlabel(r"Arg$(\lambda)$ (rad)", fontsize=label_fs)
    axPA.set_ylabel(r"$|\lambda|$", fontsize=label_fs)
    axPA.set_title("Eigenvalue angles vs magnitudes", fontsize=title_fs)
    axPA.tick_params(axis="both", labelsize=tick_fs)
    axPA.grid(True, ls=":", lw=0.4)
    for tick in axPA.get_xticklabels():
        tick.set_rotation(90)

    handles = [sc_other, sc_lead, sc_th]
    labels = ["Numerical eigenvalues", "Leading eigenvalues", r"$e^{-\alpha ik}$"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(handles),
        frameon=True,
        fontsize=legend_fs,
    )

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    for ext in formats:
        fn = f"{basename}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        print(f"Saved {fn}")
    plt.close(fig)


def plot_leading_eigenfunctions(
    galerkin_result,
    basis_matrix,
    modes=(1, -1),
    basename="spectrum_eigfuncs",
    formats=("png", "pdf"),
    dpi=1000,
):
    lam = np.asarray(galerkin_result["eigenvalues"])
    V = np.asarray(galerkin_result["eigenvectors"])
    Φ = basis_matrix
    N = Φ.shape[1]
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 3 * len(modes)), sharex=True)

    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, modes):
        target = np.exp(1j * n)
        idx = np.argmin(np.abs(lam - target))
        coeffs = V[:, idx]
        num = coeffs @ Φ
        ana = np.exp(1j * n * theta)

        num *= np.exp(-1j * np.angle(np.vdot(ana, num)))
        num /= np.abs(num).mean()

        ax.plot(theta, num.real, c="tab:red", lw=2, label="Numerical")
        ax.plot(theta, ana.real, c="k", ls="--", lw=3, label=rf"$Re(e^{{{n}i\theta}})$")
        ax.set_ylabel(r"$Re(\psi(\theta))$")
        ax.set_title(f"k = {n}   (λ ≈ {lam[idx]:.3f})")
        ax.grid(True, ls=":", lw=0.4)
        ax.legend()

    axes[-1].set_xlabel(r"$\theta$")
    fig.tight_layout()

    for ext in formats:
        fn = f"{basename}.{ext}"
        fig.savefig(fn, dpi=dpi, bbox_inches="tight")
        print(f"Saved {fn}")
    plt.close(fig)


def _orth(mat: np.ndarray, tol=1e-12):
    q, r = qr(mat, mode="reduced")
    return q[:, np.abs(np.diag(r)) > tol]


def principal_angles(B1, B2, tol=1e-12):
    Q1, Q2 = _orth(B1, tol), _orth(B2, tol)
    s = svd(Q1.T @ Q2, full_matrices=False, compute_uv=False)
    return np.arccos(np.clip(s, -1, 1))


def subspace_distance(B1, B2, metric="largest", tol=1e-12):
    theta = principal_angles(B1, B2, tol)
    return {
        "largest": theta.max(),
        "chordal": np.linalg.norm(np.sin(theta)),
        "projection": np.sin(theta.max()),
    }[metric]


def remove_low_norm_bases(basis_list, fraction=0.6):
    out = []
    for B in basis_list:
        nrm = np.linalg.norm(B, axis=1)
        out.append(B[nrm > fraction * nrm.max()])
    return out


def normalize_bases(basis_list):
    norm = []
    for B in basis_list:
        C = B.copy()
        for i, row in enumerate(C):
            v = np.linalg.norm(row)
            if v > 0:
                C[i] /= v
        norm.append(C)
    return norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoints_dir", required=True)
    ap.add_argument("-o", "--output_dir", required=True)
    args = ap.parse_args()

    cp_dir = Path(args.checkpoints_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning checkpoints …")
    subdirs = sorted(p for p in cp_dir.iterdir() if p.is_dir())

    models = []
    tins = []
    xds = []
    yds = []
    titles = []
    lambdas = []
    for sd in subdirs:
        try:
            model, t_in, _, xdata, ydata, cfg = load_model(sd, model_file="best.pth")
            ls = cfg.lambda_sparse
            models.append(model)
            tins.append(t_in)
            xds.append(xdata)
            yds.append(ydata)
            titles.append(rf"$\beta_2={ls}$")
            lambdas.append(ls)
            print(f"  loaded {sd.name} (lambda_sparse={ls})")
        except Exception as e:
            print(f"  skipping {sd.name}: {e}")

    if not models:
        print("No checkpoints - abort.")
        return

    basis_in = [
        make_peaks_positive(m.Encoder(t.reshape(-1, 2)).detach().cpu().numpy().T)
        for m, t in zip(models, tins)
    ]

    ca, sa = np.cos(ALPHA), np.sin(ALPHA)
    basis_rot = []
    for m, t in zip(models, tins):
        g = t.reshape(-1, 2).cpu().numpy()
        rot = np.c_[ca * g[:, 0] - sa * g[:, 1], sa * g[:, 0] + ca * g[:, 1]]
        rot_t = torch.as_tensor(rot, device=t.device)
        basis_rot.append(make_peaks_positive(m.Encoder(rot_t).detach().cpu().numpy().T))

    grid = tins[0].reshape(-1, 2).cpu().numpy()

    plot_bases_heatmaps(
        basis_in, grid, titles, basename=out_dir / "circle_rotation_basis_functions"
    )

    basis_pruned = remove_low_norm_bases(basis_in, 0.5)
    basis_norm = normalize_bases(basis_pruned)
    gram_norm = compute_gram_matrices(basis_norm)
    plot_gram_heatmaps(
        gram_norm, titles, basename=out_dir / "circle_rotation_gram_matrix"
    )

    plot_predictions_grid(
        xds[0],
        yds[0],
        models,
        titles,
        basename=out_dir / "circle_rotation_input_output",
    )

    print("\nSubspace distances  phi ↔ L(phi):")
    largest = []
    chordal = []
    proj = []
    for tl, B_phi, BL in zip(titles, basis_in, basis_rot):
        dL = subspace_distance(B_phi, BL, "largest")
        dC = subspace_distance(B_phi, BL, "chordal")
        dP = subspace_distance(B_phi, BL, "projection")
        largest.append(dL)
        chordal.append(dC)
        proj.append(dP)
        print(f"  {tl}: largest={dL:.5e} | chordal={dC:.5e} | projection={dP:.5e}")

    np.savez(
        out_dir / "phi_Lphi_distances.npz",
        largest=np.array(largest),
        chordal=np.array(chordal),
        projection=np.array(proj),
    )
    print("Saved phi_Lphi_distances.npz")

    idx_spec = int(np.argmax(lambdas))
    spec_model = models[idx_spec]
    phi_rows = basis_in[idx_spec]

    Lphi_rows = np.vstack([apply_operator(spec_model, row) for row in phi_rows])

    phi_cols = [row[:, None] for row in phi_rows]
    Lphi_cols = [row[:, None] for row in Lphi_rows]

    print(f"\nGalerkin spectrum from model {titles[idx_spec]}")
    result = galerkin_Lphi(
        phi_cols, Lphi_cols, inner_product_weight=None, normalize_eigenvectors=True
    )

    plot_eigenvalue_spectrum(
        result, basename=out_dir / "circle_rotation_eigenvalues_v2"
    )

    plot_leading_eigenfunctions(
        result,
        phi_rows,
        modes=(1, -1),
        basename=out_dir / "circle_rotation_eigenfunctions",
    )


if __name__ == "__main__":
    main()
