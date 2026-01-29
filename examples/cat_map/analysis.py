# analysis.py

import argparse
import datetime
import math
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.fft
import scipy.ndimage
import torch
from jax import jacfwd, jit, vmap
from matplotlib.patches import Circle
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import eig, subspace_angles

sys.path.append("../..")

from galerkin import eigs
from sabon.utils import load_model

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

jax.config.update("jax_enable_x64", True)

PI = jnp.pi
TP = 2.0 * jnp.pi


def get_phi(model, grid_size=100, normalize=False, filter_threshold=None):
    model.eval()
    with torch.no_grad():
        bases_f32 = model.Encoder(model.t_in).T.contiguous()
        w = model.trap_w_flat[None, :]
        norms_sq = (bases_f32**2 * w).sum(dim=1, keepdim=True)
        norms = torch.sqrt(norms_sq + 1e-12)

        if filter_threshold is not None:
            mask = norms.squeeze() > filter_threshold
            bases_f32 = bases_f32[mask]
            norms = norms[mask]
            total = len(mask)
            kept = mask.sum().item()
            if kept < total:
                print(
                    f"  [get_phi] Filtered basis: kept {kept}/{total} (threshold={filter_threshold})"
                )

        if normalize:
            bases_f32 = bases_f32 / norms

        phi_np = bases_f32.detach().cpu().numpy()
        reshaped = phi_np.reshape(-1, grid_size, grid_size)
        return reshaped


@jit
def T_forward(x, delta=0.01):
    x1, x2 = x[0], x[1]
    y0 = 2.0 * x1 + x2 + 2.0 * delta * jnp.cos(TP * x1)
    y1 = x1 + x2 + delta * jnp.sin(4.0 * PI * x2 + 1.0)
    return jnp.stack([y0, y1])


@jit
def F_forward(x, a=0.1, b=0.1):
    x1, x2 = x[0], x[1]
    y0 = x1 - a * jnp.sin(TP * x1)
    y1 = x2 + b * jnp.sin(TP * x2 + PI / 4.0)
    return jnp.stack([y0, y1])


@jit
def invert_T_newton(y, delta=0.01):
    guess = jnp.array([1.0 * y[0] - 1.0 * y[1], -1.0 * y[0] + 2.0 * y[1]])

    def newton_step(x):
        fx = T_forward(x, delta)
        diff = (y - fx + 0.5) % 1.0 - 0.5
        J = jacfwd(T_forward, argnums=0)(x, delta)
        delta_x = jnp.linalg.solve(J, diff)
        return jnp.mod(x + delta_x, 1.0)

    x = guess
    for _ in range(10):
        x = newton_step(x)
    return x


@jit
def invert_F_newton(y, a=0.1, b=0.1):
    def newton_step(x):
        fx = F_forward(x, a, b)
        diff = y - fx
        J = jacfwd(F_forward, argnums=0)(x, a, b)
        delta_x = jnp.linalg.solve(J, diff)
        return x + delta_x

    x = y
    for _ in range(10):
        x = newton_step(x)
    return x


@jit
def get_det_T(x, delta=0.01):
    J = jacfwd(T_forward, argnums=0)(x, delta)
    return jnp.abs(jnp.linalg.det(J))


@jit
def get_det_F(x, a=0.1, b=0.1):
    J = jacfwd(F_forward, argnums=0)(x, a, b)
    return jnp.abs(jnp.linalg.det(J))


@jit
def conjugated_inverse_and_det(x, a=0.1, b=0.1, delta=0.01):
    u = F_forward(x, a, b)
    v = invert_T_newton(u, delta)
    z = invert_F_newton(v, a, b)
    z_mod = jnp.mod(z, 1.0)
    det_F_z = get_det_F(z, a, b)
    Fz = F_forward(z, a, b)
    det_T_Fz = get_det_T(Fz, delta)
    det_F_x = get_det_F(x, a, b)
    det_Tnew_z = det_T_Fz * det_F_z / det_F_x
    return z_mod, 1.0 / det_Tnew_z


vmap_conjugated_inverse_and_det = jit(
    vmap(conjugated_inverse_and_det, in_axes=(0, None, None, None))
)


def compute_inverse_and_det_numpy(X, a=0.1, b=0.1, delta=0.01):
    X_jax = jnp.array(X)
    inv_pts, det = vmap_conjugated_inverse_and_det(X_jax, a, b, delta)
    return np.array(inv_pts), np.array(det)


def make_1d_basis(n):
    if n % 2:
        raise ValueError("n must be even")
    root2 = math.sqrt(2.0)
    out = [("c", 0, 1.0)]
    for k in range(1, n // 2):
        out += [("c", k, root2), ("s", k, root2)]
    out += [("c", n // 2, root2)]

    return out


def make_2d_basis(n):
    b1d = make_1d_basis(n)
    return [(t1, k1, t2, k2, s1 * s2) for t1, k1, s1 in b1d for t2, k2, s2 in b1d]


def compute_sobolev_norm(func_grid, order=1.0, fine_grid_factor=4):
    H, W = func_grid.shape
    H_fine, W_fine = H * fine_grid_factor, W * fine_grid_factor

    phi_fine = scipy.ndimage.zoom(
        func_grid, zoom=fine_grid_factor, order=3, mode="wrap"
    )

    kx_fine = scipy.fft.fftfreq(W_fine, d=1 / W_fine)
    ky_fine = scipy.fft.fftfreq(H_fine, d=1 / H_fine)
    KX, KY = np.meshgrid(kx_fine, ky_fine)
    K_sq = KX**2 + KY**2

    k_max_x, k_max_y = W / 2.0, H / 2.0
    mask = (np.abs(KX) < k_max_x) & (np.abs(KY) < k_max_y)

    exponent = -order / 2.0
    weights = np.zeros_like(K_sq)
    weights[mask] = (1.0 + K_sq[mask]) ** exponent

    coeffs = scipy.fft.fft2(phi_fine)
    coeffs_weighted = coeffs * weights
    phi_smooth = scipy.fft.ifft2(coeffs_weighted)

    return np.sqrt(np.mean(np.abs(phi_smooth) ** 2))


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


def savefig(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight", dpi=500)
    plt.close(fig)


def l2_l1_normalise(arr: np.ndarray) -> np.ndarray:
    a = arr / np.linalg.norm(arr, 2)
    s = a.sum()
    if s != 0:
        a = a / s
    return a


def build_transfer_operator_arrays(n_fourier, grid_size=100, a=0.1, b=0.1, delta=0.01):
    basis2d = make_2d_basis(n_fourier)
    pts = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(pts, pts, indexing="ij")
    X = np.column_stack((xx.ravel(), yy.ravel()))
    Ei = eval_basis_2d(basis2d, X)
    X_inv, det = compute_inverse_and_det_numpy(X, a, b, delta)
    Ej_raw = eval_basis_2d(basis2d, X_inv)
    Ej = Ej_raw * det[None, :]
    Ny = Nx = grid_size
    B = [Ei[i].reshape(Ny, Nx) for i in range(Ei.shape[0])]
    BT = [Ej[i].reshape(Ny, Nx) for i in range(Ej.shape[0])]
    W = np.full((Ny, Nx), 1.0 / (grid_size * grid_size))
    return B, BT, W, basis2d, X, grid_size


def plot_eigenvalues(result, ax=None, *, fig_size=(8, 6), marker="o", **scatter_kw):
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
    # turn grid on
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.scatter(
        lam[i1].real,
        lam[i1].imag,
        color="red",
        s=64,
        zorder=5,
        label="Leading eigenvalue",
    )
    ax.add_patch(
        Circle(
            (0, 0), 1.0, color="green", linestyle="--", fill=False, label="Unit circle"
        )
    )
    ax.add_patch(
        Circle(
            (0, 0),
            np.abs(lam[i2]),
            color="grey",
            linestyle=":",
            fill=False,
            label="|λ₂| circle",
        )
    )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    return fig


def compute_regularity_ratio(func_grid, order=1.0, fine_grid_factor=4):
    norm_l2 = np.sqrt(np.mean(np.abs(func_grid) ** 2))

    if norm_l2 < 1e-12:
        return 0.0

    H, W = func_grid.shape
    H_fine, W_fine = H * fine_grid_factor, W * fine_grid_factor

    phi_fine = scipy.ndimage.zoom(
        func_grid, zoom=fine_grid_factor, order=3, mode="wrap"
    )

    kx_fine = scipy.fft.fftfreq(W_fine, d=1 / W_fine)
    ky_fine = scipy.fft.fftfreq(H_fine, d=1 / H_fine)
    KX, KY = np.meshgrid(kx_fine, ky_fine)
    K_sq = KX**2 + KY**2

    # Truncation Mask
    k_max_x, k_max_y = W / 2.0, H / 2.0
    mask = (np.abs(KX) < k_max_x) & (np.abs(KY) < k_max_y)

    # Sobolev Weights (Real numbers)
    exponent = -order / 2.0
    weights = np.zeros_like(K_sq)
    weights[mask] = (1.0 + K_sq[mask]) ** exponent

    # FFT (Complex) -> Weight (Real) -> IFFT (Complex)
    coeffs = scipy.fft.fft2(phi_fine)
    coeffs_weighted = coeffs * weights
    phi_smooth = scipy.fft.ifft2(coeffs_weighted)  # DO NOT TAKE .REAL HERE

    # H^{-q} norm is the L2 norm of this smoothed function
    norm_hq = np.sqrt(np.mean(np.abs(phi_smooth) ** 2))

    return norm_hq / norm_l2


# ==========================================
# UPDATED: HIGH RES FOURIER COMPUTATION
# ==========================================


def compute_high_res_srb(
    n_fourier=100,
    grid_size=200,
    a=0.1,
    b=0.1,
    delta=0.01,
    load_path=None,
    save_path=None,
):
    target_grid_size = 100

    if load_path is not None and os.path.exists(load_path):
        print(f"Loading High-Res SRB from {load_path}...")
        try:
            srb_density = np.load(load_path)
            if srb_density.shape == (target_grid_size, target_grid_size):
                print("  Successfully loaded.")
                return srb_density
            else:
                print(f"  Loaded shape {srb_density.shape} mismatch. Recomputing...")
        except Exception as e:
            print(f"  Failed to load: {e}. Recomputing...")

    print(
        f"Computing High-Res SRB Ground Truth (Fourier N={n_fourier} on {grid_size}x{grid_size})..."
    )

    basis2d = make_2d_basis(n_fourier)

    # High-Res Grid (200x200) for Galerkin Integration
    pts = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(pts, pts, indexing="ij")
    X_grid = np.column_stack((xx.ravel(), yy.ravel()))

    # Build Matrices
    print("  Computing conjugated inverse...")
    X_inv, det = compute_inverse_and_det_numpy(X_grid, a, b, delta)

    print("  Evaluating basis (High Res)...")
    B = eval_basis_2d(basis2d, X_grid)
    BT_raw = eval_basis_2d(basis2d, X_inv)
    BT = BT_raw * det[None, :]
    w = 1.0 / X_grid.shape[0]

    print("  Solving eigenproblem...")
    G = B @ (BT * w).T
    vals, vecs = eig(G)

    # Sort eigenvalues
    idx_sorted = np.argsort(-np.abs(vals))

    # Create 100x100 Evaluation Grid
    pts_eval = (np.arange(target_grid_size) + 0.5) / target_grid_size
    px, py = np.meshgrid(pts_eval, pts_eval, indexing="ij")
    X_eval = np.column_stack((px.ravel(), py.ravel()))

    # Evaluate Basis on 100x100
    B_eval = eval_basis_2d(basis2d, X_eval)

    # Reconstruct Final SRB Density (Real part of leading mode)
    leading_idx = idx_sorted[0]
    srb_density = (vecs[:, leading_idx].real @ B_eval).reshape(
        target_grid_size, target_grid_size
    )

    # Normalize
    srb_density = l2_l1_normalise(srb_density)
    print("  High-Res SRB computation done.")

    # Save 100x100 result
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, srb_density)
            print(f"  Saved High-Res SRB to {save_path}")
        except Exception as e:
            print(f"  Warning: Could not save SRB to {save_path}: {e}")

    return srb_density


def compute_fourier_eigenproblem(n_fourier, grid_size, a=0.1, b=0.1, delta=0.01):
    """Compute Galerkin eigenproblem for Fourier basis. Returns (eigenvalues, eigenvectors, basis_on_grid)."""
    basis2d = make_2d_basis(n_fourier)

    pts = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(pts, pts, indexing="ij")
    X_grid = np.column_stack((xx.ravel(), yy.ravel()))

    X_inv, det = compute_inverse_and_det_numpy(X_grid, a, b, delta)

    B = eval_basis_2d(basis2d, X_grid)
    BT = eval_basis_2d(basis2d, X_inv) * det[None, :]
    w = 1.0 / X_grid.shape[0]

    G = B @ (BT * w).T
    vals, vecs = eig(G)

    return vals, vecs, B


def compute_h_minus_1_angles(basis1, basis2, fine_grid_factor=4, order=1.0):
    H, W = basis1[0].shape
    H_fine, W_fine = H * fine_grid_factor, W * fine_grid_factor
    kx_fine = scipy.fft.fftfreq(W_fine, d=1 / W_fine)
    ky_fine = scipy.fft.fftfreq(H_fine, d=1 / H_fine)
    KX, KY = np.meshgrid(kx_fine, ky_fine)
    k_max_x, k_max_y = W / 2.0, H / 2.0
    mask = (np.abs(KX) < k_max_x) & (np.abs(KY) < k_max_y)

    K_sq = KX**2 + KY**2
    exponent = -order / 2.0
    weights = np.zeros_like(K_sq)
    weights[mask] = (1.0 + K_sq[mask]) ** exponent

    def get_smoothed(basis):
        vecs = []
        for phi in basis:
            pf = scipy.ndimage.zoom(phi, zoom=fine_grid_factor, order=3, mode="wrap")
            coeffs = scipy.fft.fft2(pf)
            coeffs_w = coeffs * weights
            ps = scipy.fft.ifft2(coeffs_w).real
            vecs.append(ps.ravel())
        return np.stack(vecs, axis=1)

    A_spatial = get_smoothed(basis1)
    B_spatial = get_smoothed(basis2)
    return np.sort(subspace_angles(A_spatial, B_spatial))


def compute_h_minus1_inner_product(f, g, grid_size):
    """Compute <f, g>_{H^{-1}} via FFT."""
    fft_f = np.fft.fft2(f)
    fft_g = np.fft.fft2(g)

    freq_x = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    freq_y = np.fft.fftfreq(grid_size, d=1.0 / grid_size)
    kx, ky = np.meshgrid(freq_x, freq_y, indexing="ij")
    k_sq = kx**2 + ky**2
    weights = 1.0 / (1.0 + k_sq)  # Matches compute_sobolev_norm with order=1.0

    return np.real(np.sum(weights * fft_f * np.conj(fft_g))) / grid_size**2


def apply_transfer_operator(func_grid, inv_pts_grid, det_grid, grid_coords):
    """Apply transfer operator K: (Kf)(x) = f(T^{-1}(x)) * |det(DT^{-1}(x))|"""
    if np.iscomplexobj(func_grid):
        real_part = apply_transfer_operator(
            func_grid.real, inv_pts_grid, det_grid, grid_coords
        )
        imag_part = apply_transfer_operator(
            func_grid.imag, inv_pts_grid, det_grid, grid_coords
        )
        return real_part + 1j * imag_part

    interp = RegularGridInterpolator(
        (grid_coords, grid_coords),
        func_grid,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )
    return interp(inv_pts_grid) * det_grid


def compute_eigenfunction_residual(
    func_grid, eigenvalue, inv_pts_grid, det_grid, grid_coords
):
    """Compute L² residual: ||Kg - λg||² / ||g||²"""
    Kg = apply_transfer_operator(func_grid, inv_pts_grid, det_grid, grid_coords)
    residual = Kg - eigenvalue * func_grid

    residual_norm_sq = np.mean(np.abs(residual) ** 2)
    func_norm_sq = np.mean(np.abs(func_grid) ** 2)

    return residual_norm_sq / func_norm_sq


def project_h_minus1(basis_flat, target_vec, grid_size):
    n_basis = basis_flat.shape[0]
    basis_2d = [b.reshape(grid_size, grid_size) for b in basis_flat]
    target_2d = target_vec.reshape(grid_size, grid_size)

    # Build H^{-1} Gram matrix
    M = np.zeros((n_basis, n_basis))
    for i in range(n_basis):
        for j in range(i, n_basis):
            M[i, j] = compute_h_minus1_inner_product(
                basis_2d[i], basis_2d[j], grid_size
            )
            M[j, i] = M[i, j]

    # H^{-1} inner products with target
    b = np.array(
        [
            compute_h_minus1_inner_product(basis_2d[i], target_2d, grid_size)
            for i in range(n_basis)
        ]
    )

    coeff = np.linalg.solve(M, b)
    proj = (coeff @ basis_flat).reshape(grid_size, grid_size)
    return proj.T


def run_spectrum(model, t_in, device, out_dir, n_fourier=18, a=0, b=0, delta=0.01):
    grid_size = 100

    phi = get_phi(model, grid_size)
    n_basis = phi.shape[0]
    phi_flat = phi.reshape(n_basis, -1)

    pts = (np.arange(grid_size) + 0.5) / grid_size
    xx, yy = np.meshgrid(pts, pts, indexing="ij")
    X_grid = np.column_stack((xx.ravel(), yy.ravel()))

    inv_pts, det = compute_inverse_and_det_numpy(X_grid, a, b, delta)
    inv_pts_grid = inv_pts.reshape(grid_size, grid_size, 2)
    det_grid = det.reshape(grid_size, grid_size)
    grid_coords = np.linspace(0, 1, grid_size, endpoint=False) + 0.5 / grid_size

    Lphi_list = []
    for i in range(n_basis):
        interp = RegularGridInterpolator(
            (grid_coords, grid_coords),
            phi[i],
            method="cubic",
            bounds_error=False,
            fill_value=None,
        )
        Lphi_list.append(interp(inv_pts_grid) * det_grid)
    Lphi = np.array(Lphi_list)
    w = 1.0 / (grid_size**2)

    Lphi_flat = Lphi.reshape(n_basis, -1)
    G_ml = phi_flat @ (Lphi_flat * w).T
    _, vecs_ml = eig(G_ml)

    results_ml = eigs(model, t_in)
    vals_ml = results_ml["eigenvalues"]
    fig_ev = plot_eigenvalues(results_ml)
    savefig(fig_ev, os.path.join(out_dir, "anosov_eigenvalues"))

    idx_sorted = np.argsort(-np.abs(vals_ml))

    leading_idx = idx_sorted[0]
    coeff_ml = vecs_ml[:, leading_idx].real

    eig_ml = (coeff_ml @ phi_flat).reshape(grid_size, grid_size).T
    M = results_ml["M"]
    A = results_ml["A"]  # A = G @ M
    G_learned = M @ A  # This gives M @ G @ M ≈ G_true

    # Eigendecomposition (same as true dynamics)
    vals_learned, vecs_learned = eig(G_learned)
    idx_sorted_learned = np.argsort(-np.abs(vals_learned))
    leading_idx_learned = idx_sorted_learned[0]
    learned_coeff = vecs_learned[:, leading_idx_learned].real
    eig_ml_learned = (learned_coeff @ phi_flat).reshape(grid_size, grid_size).T
    if a == b == 0:
        ground_truth_path = "/srv/scratch/z5547452/work/SABON/examples/cat_map/ground_truths/srb_density_high_unconjugated.npy"
        fourier_approx_path = f"/srv/scratch/z5547452/work/SABON/examples/cat_map/ground_truths/srb_density_{n_fourier}_unconjugated.npy"
    else:
        ground_truth_path = "/srv/scratch/z5547452/work/SABON/examples/cat_map/ground_truths/srb_density_high_conjugated.npy"
        fourier_approx_path = f"/srv/scratch/z5547452/work/SABON/examples/cat_map/ground_truths/srb_density_{n_fourier}_conjugated.npy"

    gt_srb = compute_high_res_srb(
        100,
        200,
        a,
        b,
        delta,
        load_path=ground_truth_path,
        save_path=None,
    )

    vals_four, vecs_four, B_four = compute_fourier_eigenproblem(
        n_fourier, grid_size, a, b, delta
    )
    idx_sorted_four = np.argsort(-np.abs(vals_four))
    leading_idx_four = idx_sorted_four[0]
    coeff_four = vecs_four[:, leading_idx_four].real
    eig_four = (coeff_four @ B_four).reshape(grid_size, grid_size).T

    mu_vec_ij = gt_srb.T.ravel(order="C")
    w_uniform = np.full_like(mu_vec_ij, 1.0 / mu_vec_ij.size)

    basis2d = make_2d_basis(n_fourier)
    phi_four_flat = eval_basis_2d(basis2d, X_grid)

    def project_and_restore(basis_flat, w, target_vec):
        norms = np.linalg.norm(basis_flat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Bn = basis_flat / norms
        M = (Bn * w) @ Bn.T
        b = (Bn * w) @ target_vec
        coeff = np.linalg.solve(M, b)
        proj_ij = (coeff @ Bn).reshape(grid_size, grid_size)
        return proj_ij.T

    proj_ml = project_and_restore(phi_flat, w_uniform, mu_vec_ij)
    proj_four = project_and_restore(phi_four_flat, w_uniform, mu_vec_ij)

    proj_ml_h1 = project_h_minus1(phi_flat, mu_vec_ij, grid_size)
    proj_four_h1 = project_h_minus1(phi_four_flat, mu_vec_ij, grid_size)

    # 6. PLOT 1: SRB RECONSTRUCTION
    gt_norm = l2_l1_normalise(gt_srb.copy())
    eig_ml_norm = l2_l1_normalise(eig_ml.copy())
    eig_four_norm = l2_l1_normalise(eig_four.copy())

    vmin = min(gt_norm.min(), eig_ml_norm.min(), eig_four_norm.min()) * 10000
    vmax = max(gt_norm.max(), eig_ml_norm.max(), eig_four_norm.max()) * 10000
    extent = (0, 1, 0, 1)

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    titles1 = [
        "Ground-truth SRB measure",
        "Reconstructed measure (SABON)",
        "Reconstructed measure (Fourier)",
    ]
    data1 = [gt_norm, eig_ml_norm, eig_four_norm]
    TICKS = [0, 0.5, 1]
    TICK_LABELS = ["0", "0.5", "1"]

    tick_fs = 11

    for ax, arr, title in zip(axes1, data1, titles1):
        # set title fontsize to 14
        ax.set_title(title, fontsize=14)
        ax.set_xticks(TICKS, TICK_LABELS, fontsize=tick_fs)
        ax.set_yticks(TICKS, TICK_LABELS, fontsize=tick_fs)
        im = ax.imshow(
            arr * 10000,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap="Blues",
            extent=extent,
        )
        ax.set_aspect("equal")

    # add label "Normalised value" to colorbar with fontsize 12, from top to bottom

    cbar = fig1.colorbar(im, ax=axes1.ravel().tolist(), location="right", shrink=0.8)
    cbar.set_label("Normalised value", va="bottom", fontsize=12, labelpad=14)
    savefig(fig1, os.path.join(out_dir, "anosov_srb_eigenvectors"))

    # PROJECTED GROUND TRUTH
    proj_ml_norm = l2_l1_normalise(proj_ml.copy())
    proj_four_norm = l2_l1_normalise(proj_four.copy())

    proj_ml_h1_norm = l2_l1_normalise(proj_ml_h1.copy())
    proj_four_h1_norm = l2_l1_normalise(proj_four_h1.copy())

    vmin_p = min(gt_norm.min(), proj_ml_norm.min(), proj_four_norm.min()) * 10000
    vmax_p = max(gt_norm.max(), proj_ml_norm.max(), proj_four_norm.max()) * 10000

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    titles2 = [
        "Ground-truth SRB measure",
        "Projected GT (SABON)",
        "Projected GT (Fourier)",
    ]
    data2 = [gt_norm, proj_ml_norm, proj_four_norm]

    for ax, arr, title in zip(axes2, data2, titles2):
        im2 = ax.imshow(
            arr * 10000,
            vmin=vmin_p,
            vmax=vmax_p,
            origin="lower",
            cmap="Blues",
            extent=extent,
        )
        ax.set_title(title, fontsize=14)
        ax.set_aspect("equal")

    fig2.colorbar(im2, ax=axes2.ravel().tolist(), location="right", shrink=0.8)
    savefig(fig2, os.path.join(out_dir, "anosov_srb_projections"))

    # ============ ERRORS (RAW, NO POST-NORMALIZATION) ============
    gt_raw = gt_srb.copy()
    gt_vec = gt_raw.ravel()
    gt_norm_l2 = np.linalg.norm(gt_vec)
    gt_norm_h1 = compute_sobolev_norm(gt_raw, order=1.0)

    # Projections (raw)
    proj_ml_vec = proj_ml.ravel()
    proj_four_vec = proj_four.ravel()

    # Reconstructions (raw)
    eig_ml_vec = eig_ml.ravel()
    eig_four_vec = eig_four.ravel()

    # --- L2 ERRORS ---
    # Projection
    proj_err_ml_l2 = np.linalg.norm(proj_ml_vec - gt_vec) / gt_norm_l2
    proj_err_four_l2 = np.linalg.norm(proj_four_vec - gt_vec) / gt_norm_l2

    # Approximation (optimal scaling: c* = <g, μ> / <g, g>)
    c_ml_l2 = np.dot(eig_ml_vec, gt_vec) / np.dot(eig_ml_vec, eig_ml_vec)
    c_four_l2 = np.dot(eig_four_vec, gt_vec) / np.dot(eig_four_vec, eig_four_vec)

    approx_err_ml_l2 = np.linalg.norm(c_ml_l2 * eig_ml_vec - gt_vec) / gt_norm_l2
    approx_err_four_l2 = np.linalg.norm(c_four_l2 * eig_four_vec - gt_vec) / gt_norm_l2

    # --- H^{-1} ERRORS ---
    # Projection (using H^{-1} projections)
    diff_ml_h1 = proj_ml_h1 - gt_raw
    diff_four_h1 = proj_four_h1 - gt_raw

    proj_err_ml_h1 = compute_sobolev_norm(diff_ml_h1, order=1.0) / gt_norm_h1
    proj_err_four_h1 = compute_sobolev_norm(diff_four_h1, order=1.0) / gt_norm_h1

    # Approximation (optimal scaling in H^{-1}: c* = <g, μ>_{H^{-1}} / <g, g>_{H^{-1}})
    eig_ml_2d = eig_ml
    eig_four_2d = eig_four

    c_ml_h1 = compute_h_minus1_inner_product(
        eig_ml_2d, gt_raw, grid_size
    ) / compute_h_minus1_inner_product(eig_ml_2d, eig_ml_2d, grid_size)
    c_four_h1 = compute_h_minus1_inner_product(
        eig_four_2d, gt_raw, grid_size
    ) / compute_h_minus1_inner_product(eig_four_2d, eig_four_2d, grid_size)

    diff_ml_approx_h1 = c_ml_h1 * eig_ml_2d - gt_raw
    diff_four_approx_h1 = c_four_h1 * eig_four_2d - gt_raw

    approx_err_ml_h1 = compute_sobolev_norm(diff_ml_approx_h1, order=1.0) / gt_norm_h1
    approx_err_four_h1 = (
        compute_sobolev_norm(diff_four_approx_h1, order=1.0) / gt_norm_h1
    )

    # --- LEARNED OPERATOR APPROXIMATION ERRORS (hat{mu}_B) ---
    eig_ml_learned_vec = eig_ml_learned.ravel()

    eig_ml_learned_norm = l2_l1_normalise(eig_ml_learned.copy())

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(
        eig_ml_learned_norm * 10000,
        origin="lower",
        cmap="Blues",
        extent=(0, 1, 0, 1),
    )
    ax.set_title(r"SABON $\hat{\mu}_{\mathcal{B}}$ (learned operator)", fontsize=14)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, os.path.join(out_dir, "anosov_srb_learned"))

    # L2
    c_ml_learned_l2 = np.dot(eig_ml_learned_vec, gt_vec) / np.dot(
        eig_ml_learned_vec, eig_ml_learned_vec
    )
    approx_err_ml_learned_l2 = (
        np.linalg.norm(c_ml_learned_l2 * eig_ml_learned_vec - gt_vec) / gt_norm_l2
    )

    # H^-1
    c_ml_learned_h1 = compute_h_minus1_inner_product(
        eig_ml_learned, gt_raw, grid_size
    ) / compute_h_minus1_inner_product(eig_ml_learned, eig_ml_learned, grid_size)
    diff_ml_learned_approx_h1 = c_ml_learned_h1 * eig_ml_learned - gt_raw
    approx_err_ml_learned_h1 = (
        compute_sobolev_norm(diff_ml_learned_approx_h1, order=1.0) / gt_norm_h1
    )

    # --- PRINT ---
    print("\n" + "=" * 65)
    print("  ERRORS (RAW, NO POST-NORMALIZATION)")
    print("=" * 65)
    print(f"  {'Metric':<30} | {'SABON':<15} | {'Fourier':<15}")
    print("-" * 65)
    print(
        f"  {'L2 Projection':<30} | {proj_err_ml_l2:<15.6e} | {proj_err_four_l2:<15.6e}"
    )
    print(
        f"  {'L2 Approx (true dynamics)':<30} | {approx_err_ml_l2:<15.6e} | {approx_err_four_l2:<15.6e}"
    )
    print(
        f"  {'L2 Approx (learned op.)':<30} | {approx_err_ml_learned_l2:<15.6e} | {'---':<15}"
    )
    print(
        f"  {'H^-1 Projection':<30} | {proj_err_ml_h1:<15.6e} | {proj_err_four_h1:<15.6e}"
    )
    print(
        f"  {'H^-1 Approx (true dynamics)':<30} | {approx_err_ml_h1:<15.6e} | {approx_err_four_h1:<15.6e}"
    )
    print(
        f"  {'H^-1 Approx (learned op.)':<30} | {approx_err_ml_learned_h1:<15.6e} | {'---':<15}"
    )
    print("=" * 65)

    # Sanity checks
    assert proj_err_ml_l2 <= approx_err_ml_l2 + 1e-10, "SABON L2: proj > approx!"
    assert proj_err_four_l2 <= approx_err_four_l2 + 1e-10, "Fourier L2: proj > approx!"
    assert proj_err_ml_h1 <= approx_err_ml_h1 + 1e-10, "SABON H^-1: proj > approx!"
    assert proj_err_four_h1 <= approx_err_four_h1 + 1e-10, (
        "Fourier H^-1: proj > approx!"
    )

    indices, ratios_h1, ratios_h2, lambdas = [], [], [], []

    vals_ml = results_ml["eigenvalues"]
    vecs_ml = results_ml["eigenvectors"]
    idx_sorted = np.argsort(-np.abs(vals_ml))
    top_k = 20
    print(f"\n  Analyzing SABON Regularity of Top {top_k} Eigenvectors (Complex)...")
    print(f"  {'Index':<6} | {'|Lam|':<8} | {'Ratio H^-1':<12} | {'Ratio H^-2':<12}")
    print("-" * 50)
    for k in range(min(top_k, len(idx_sorted))):
        idx = idx_sorted[k]
        val_mag = np.abs(vals_ml[idx])

        func_grid_complex = (vecs_ml[:, idx] @ phi_flat).reshape(grid_size, grid_size)
        nrm = np.linalg.norm(func_grid_complex)
        if nrm > 0:
            func_grid_complex /= nrm

        r1 = compute_regularity_ratio(func_grid_complex, order=1.0)
        r2 = compute_regularity_ratio(func_grid_complex, order=2.0)

        indices.append(k)
        lambdas.append(val_mag)
        ratios_h1.append(r1)
        ratios_h2.append(r2)

        print(f"  {k:<6} | {val_mag:<8.4f} | {r1:<12.4f} | {r2:<12.4f}")

    # ===== FOURIER RESIDUAL COMPUTATION =====
    print("\n" + "=" * 65)
    print("  FOURIER EIGENFUNCTION RESIDUALS")
    print("=" * 65)

    idx_sorted_four = np.argsort(-np.abs(vals_four))

    print(f"  {'Rank':<6} {'|λ|':<12} {'λ':<28} {'Residual':<15}")
    print("-" * 65)

    fourier_residuals = []
    for rank in range(10):
        idx = idx_sorted_four[rank]
        lam = vals_four[idx]
        coeff = vecs_four[:, idx]

        eig_func = (coeff @ B_four).reshape(grid_size, grid_size)

        res = compute_eigenfunction_residual(
            eig_func, lam, inv_pts_grid, det_grid, grid_coords
        )
        fourier_residuals.append(res)

        print(f"  {rank:<6} {np.abs(lam):<12.6f} {lam:<28} {res:<15.6e}")

    print("=" * 65)

    return {
        "n_fourier": n_fourier,
    }


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
        gt_np.T, cmap="bwr", origin="lower", vmin=clim[0], vmax=clim[1], extent=EXTENT
    )
    axes[1].set_title("Ground truth", fontsize=title_fs)
    style(axes[1])

    axes[2].imshow(
        pred_np.T, cmap="bwr", origin="lower", vmin=clim[0], vmax=clim[1], extent=EXTENT
    )
    axes[2].set_title("Predicted", fontsize=title_fs)
    style(axes[2])

    mappable = axes[1].images[0]
    cbar = fig.colorbar(
        mappable, ax=axes.ravel().tolist(), shrink=0.85, location="right"
    )
    cbar.ax.set_ylabel("Observable value", va="bottom", fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs)

    savefig(fig, os.path.join(out_dir, "anosov_input_output"))
    return mean_rel_l2_error


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
    cbar.set_label(r"$M_{kj}$", fontsize=20, labelpad=14)
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
            norm_p.T, cmap="bwr", origin="lower", vmin=-vmax, vmax=vmax, extent=EXTENT
        )
        style_axes(ax, show_frame=False)
        ax.set_title(rf"$\phi_{{{sel[i]}}}$", fontsize=22)

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), location="right", shrink=0.8, pad=0.02
    )
    cbar.set_label("Normalised basis value", labelpad=14, fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    savefig(fig, os.path.join(out_dir, "anosov_basis"))


def main():
    p = argparse.ArgumentParser(description="SABON Analysis")
    p.add_argument("--checkpoint_dir", required=True, help="Checkpoint dir")
    p.add_argument("--output_dir", required=True, help="Output dir")
    args = p.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.output_dir, ts)
    for sub in ("angle", "plots", "spectrum"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    model, t_in, _, x_data, y_data, config = load_model(
        args.checkpoint_dir, model_file="best.pth"
    )
    device = config.device

    all_metrics = {}

    print("\n--- Computing Spectrum ---")
    srb_res = run_spectrum(
        model,
        t_in,
        device,
        os.path.join(root, "spectrum"),
    )
    all_metrics.update(srb_res)

    print("Computing basis and gram matrix visualisations...")
    plot_basis_and_gram(model, t_in, os.path.join(root, "plots"))

    print("\n--- Computing Prediction Error ---")
    mean_rel_l2_error = plot_single_prediction(
        model, x_data, y_data, os.path.join(root, "plots")
    )

    print("-------------------------------------------------------------")
    print(f"--> Test set mean relative L2 error: {mean_rel_l2_error:.6e}")
    print("-------------------------------------------------------------")

    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
