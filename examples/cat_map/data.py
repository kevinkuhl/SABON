# data.py

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jacfwd, jit, vmap

# Enable float64
jax.config.update("jax_enable_x64", True)

PI = jnp.pi
TP = 2.0 * jnp.pi

# ==========================================
# Map Definitions (JAX)
# ==========================================


@jit
def T_forward(x, delta=0.01):
    """Perturbed cat map (forward, no mod)."""
    x1, x2 = x[0], x[1]
    y0 = 2.0 * x1 + x2 + 2.0 * delta * jnp.cos(TP * x1)
    y1 = x1 + x2 + delta * jnp.sin(4.0 * PI * x2 + 1.0)
    return jnp.stack([y0, y1])


@jit
def T_forward_mod(x, delta=0.01):
    """Perturbed cat map (forward, with mod)."""
    return jnp.mod(T_forward(x, delta), 1.0)


@jit
def F_forward(x, a=0.1, b=0.1):
    """Conjugacy map F: x -> x + a*sin(2*pi*x) for each component."""
    x1, x2 = x[0], x[1]
    y0 = x1 - a * jnp.sin(TP * x1)
    y1 = x2 + b * jnp.sin(TP * x2 + PI / 4.0)
    return jnp.stack([y0, y1])


# ==========================================
# Inverse Functions (Newton's Method)
# ==========================================


@jit
def invert_T_newton(y, delta=0.01):
    """Invert the perturbed cat map using Newton's method."""
    # Initial guess from linear inverse
    guess = jnp.array([1.0 * y[0] - 1.0 * y[1], -1.0 * y[0] + 2.0 * y[1]])

    def newton_step(x):
        fx = T_forward(x, delta)
        # Periodic residual
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
    """Invert the conjugacy map F using Newton's method."""

    def newton_step(x):
        fx = F_forward(x, a, b)
        diff = y - fx
        J = jacfwd(F_forward, argnums=0)(x, a, b)
        delta_x = jnp.linalg.solve(J, diff)
        return x + delta_x

    x = y  # Initial guess
    for _ in range(10):
        x = newton_step(x)
    return x


# ==========================================
# Determinant Functions
# ==========================================


@jit
def get_det_T(x, delta=0.01):
    """Jacobian determinant of T at x."""
    J = jacfwd(T_forward, argnums=0)(x, delta)
    return jnp.abs(jnp.linalg.det(J))


@jit
def get_det_F(x, a=0.1, b=0.1):
    """Jacobian determinant of F at x."""
    J = jacfwd(F_forward, argnums=0)(x, a, b)
    return jnp.abs(jnp.linalg.det(J))


# ==========================================
# Conjugated Map Inverse & Determinant
# ==========================================


@jit
def conjugated_inverse_and_det(x, a=0.1, b=0.1, delta=0.01):
    """
    Compute the inverse of Tnew = F⁻¹ ∘ T ∘ F and 1/|det(J_Tnew)|.

    Returns (z, det_factor) where:
        z = Tnew^{-1}(x)
        det_factor = 1 / |det(J_Tnew(z))|
    """
    # Step 1: u = F(x)
    u = F_forward(x, a, b)

    # Step 2: v = T⁻¹(u) = T⁻¹(F(x))
    v = invert_T_newton(u, delta)

    # Step 3: z = F⁻¹(v) = F⁻¹(T⁻¹(F(x))) = Tnew⁻¹(x)
    z = invert_F_newton(v, a, b)
    z_mod = jnp.mod(z, 1.0)

    # Compute det(J_Tnew(z))
    det_F_z = get_det_F(z, a, b)
    Fz = F_forward(z, a, b)
    det_T_Fz = get_det_T(Fz, delta)
    det_F_x = get_det_F(x, a, b)

    det_Tnew_z = det_T_Fz * det_F_z / det_F_x

    # Return 1/det for transfer operator: P(f)(x) = f(T^{-1}(x)) * (1/det)
    return z_mod, 1.0 / det_Tnew_z


# Vectorized versions
vmap_conjugated_inverse_and_det = vmap(
    conjugated_inverse_and_det, in_axes=(0, None, None, None)
)

# ==========================================
# Fourier Basis
# ==========================================


def make_1d_basis(m):
    """Create 1D Fourier basis up to order m."""
    r2 = np.sqrt(2.0)
    b = [("c", 0, 1.0)]
    for k in range(1, m + 1):
        b += [("c", k, r2), ("s", k, r2)]
    return b


def make_2d_basis(m):
    """Create 2D tensor product Fourier basis."""
    b1 = make_1d_basis(m)
    return [(t1, k1, t2, k2, s1 * s2) for t1, k1, s1 in b1 for t2, k2, s2 in b1]


def eval_basis_2d_torus(B, x1, y1, x2, y2):
    """
    Evaluate 2D basis on embedded torus coordinates.
    (x1, y1) and (x2, y2) are cos/sin pairs for each dimension.
    """
    tp = 2 * np.pi
    t1 = (np.arctan2(y1, x1) % (2 * np.pi)) / tp
    t2 = (np.arctan2(y2, x2) % (2 * np.pi)) / tp
    v = np.empty((len(B), t1.size))
    for i, (a, kA, b, kB, s) in enumerate(B):
        fA = np.cos if a == "c" else np.sin
        fB = np.cos if b == "c" else np.sin
        v[i] = s * fA(tp * kA * t1) * fB(tp * kB * t2)
    return v


def eval_basis_2d_flat(B, pts):
    """
    Evaluate 2D basis on flat [0,1]^2 coordinates.
    pts: array of shape (N, 2) with values in [0, 1]
    """
    tp = 2 * np.pi
    t1, t2 = pts[:, 0], pts[:, 1]
    v = np.empty((len(B), t1.size))
    for i, (a, kA, b, kB, s) in enumerate(B):
        fA = np.cos if a == "c" else np.sin
        fB = np.cos if b == "c" else np.sin
        v[i] = s * fA(tp * kA * t1) * fB(tp * kB * t2)
    return v


def random_linear_combinations(B, n, rng):
    """Generate n random linear combinations of basis functions."""
    nB = len(B)
    C = rng.uniform(-1, 1, (n, nB))

    def mk(c):
        return lambda x1, y1, x2, y2: (
            c @ eval_basis_2d_torus(B, x1, y1, x2, y2)
        ).ravel()

    return [mk(c) for c in C]


def midpoint_grid(n):
    """Create midpoint grid for numerical integration."""
    p = (np.arange(n) + 0.5) / n
    th = 2 * np.pi * p
    return p, np.cos(th), np.sin(th)


# ==========================================
# K-th Iterate Computation
# ==========================================


def compute_k_iterate(flat_grid_pts, k, a, b, delta):
    """
    Compute the k-th iterate of the inverse map and cumulative determinant.

    For the transfer operator L^k, we need:
        L^k(f)(x) = f(T^{-k}(x)) * prod_{i=0}^{k-1} (1/|det J_T(T^{-i-1}(x))|)

    Args:
        flat_grid_pts: Grid points (N, 2)
        k: Number of iterations
        a, b, delta: Map parameters

    Returns:
        inv_pts_k: T^{-k}(x) for each grid point
        det_k: Cumulative determinant factor
    """
    X_grid_jax = jnp.array(flat_grid_pts)

    # First iteration
    inv_pts, det_cumul = vmap_conjugated_inverse_and_det(X_grid_jax, a, b, delta)
    inv_pts = np.array(inv_pts)
    det_cumul = np.array(det_cumul)

    # Subsequent iterations
    for i in range(1, k):
        print(f"  Computing iterate {i + 1}/{k}...")
        inv_pts_jax, det_step_jax = vmap_conjugated_inverse_and_det(
            jnp.array(inv_pts), a, b, delta
        )
        inv_pts = np.array(inv_pts_jax)
        det_cumul = det_cumul * np.array(det_step_jax)

    return inv_pts, det_cumul


def evaluate_transfer_operator(F, inv_pts, det, npix):
    """
    Apply transfer operator to functions F using precomputed inverse and determinant.

    Args:
        F: List of function callables
        inv_pts: Inverse points (N, 2) in [0,1]^2
        det: Determinant factors (N,)
        npix: Grid resolution

    Returns:
        ydata: Transformed functions (nf, npix, npix)
    """
    # Convert inverse points to torus embedding
    t1 = 2 * np.pi * inv_pts[:, 0]
    t2 = 2 * np.pi * inv_pts[:, 1]
    inv_x, inv_y = np.cos(t1), np.sin(t1)
    inv_u, inv_v = np.cos(t2), np.sin(t2)

    nf = len(F)
    ydata = np.empty((nf, npix, npix))
    for j, f in enumerate(F):
        val_at_inv = f(inv_x, inv_y, inv_u, inv_v)
        to_val = val_at_inv * det
        ydata[j] = to_val.reshape(npix, npix)

    return ydata


# ==========================================
# Dataset Building
# ==========================================


def build_datasets(
    nf, npix, mx, k_iterate=1, a=0.1, b=0.1, delta=0.01, seed=42, save_dir=None
):
    """
    Build training datasets for the transfer operator.

    Args:
        nf: Number of functions to generate
        npix: Grid resolution per dimension
        mx: Maximum Fourier order
        k_iterate: Number of iterations for output data (default 1)
        a, b: Conjugacy parameters
        delta: Perturbation parameter for cat map
        seed: Random seed
        save_dir: Directory to save data (optional)

    Returns:
        B: Basis specification
        xdata: Input functions evaluated on grid
        ydata: Transfer operator applied once (for operator loss)
        ydata_k: Transfer operator applied k times (for output autoencoder loss)
        pts: Grid points
    """
    rng = np.random.default_rng(seed)

    print(f"Map parameters: delta={delta}, a={a}, b={b}")
    print(f"Grid: {npix}x{npix}, Functions: {nf}, Max order: {mx}")
    print(f"K-iterate: {k_iterate}")

    # Create basis and random functions
    B = make_2d_basis(mx)
    F = random_linear_combinations(B, nf, rng)
    print(f"Basis size: {len(B)}")

    # Create 4D embedded torus grid for function evaluation
    pts, x1, y1 = midpoint_grid(npix)
    _, x2, y2 = midpoint_grid(npix)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    Y1, Y2 = np.meshgrid(y1, y2, indexing="ij")
    grid4d = np.stack([X1, Y1, X2, Y2], -1)

    xf, yf, uf, vf = (grid4d[..., i].ravel() for i in range(4))

    # Evaluate input functions
    xdata = np.empty((nf, npix, npix))
    for j, f in enumerate(F):
        xdata[j] = f(xf, yf, uf, vf).reshape(npix, npix)

    # Create flat grid for inverse computation
    p_grid = (np.arange(npix) + 0.5) / npix
    gx, gy = np.meshgrid(p_grid, p_grid, indexing="ij")
    flat_grid_pts = np.stack((gx.ravel(), gy.ravel()), axis=1)

    # Compute 1-step inverse and determinant (always needed for operator loss)
    print("Computing 1-step conjugated inverse (JAX)...")
    X_grid_jax = jnp.array(flat_grid_pts)
    inv_pts_1, det_1 = vmap_conjugated_inverse_and_det(X_grid_jax, a, b, delta)
    inv_pts_1 = np.array(inv_pts_1)
    det_1 = np.array(det_1)

    # Apply 1-step transfer operator
    print("Evaluating 1-step transfer operator...")
    ydata = evaluate_transfer_operator(F, inv_pts_1, det_1, npix)

    # Compute k-step inverse and determinant if k > 1
    if k_iterate > 1:
        print(f"Computing {k_iterate}-step conjugated inverse (JAX)...")
        inv_pts_k, det_k = compute_k_iterate(flat_grid_pts, k_iterate, a, b, delta)

        print(f"Evaluating {k_iterate}-step transfer operator...")
        ydata_k = evaluate_transfer_operator(F, inv_pts_k, det_k, npix)
    else:
        # k=1 case: ydata_k is the same as ydata
        ydata_k = ydata

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "xdata.npy"), xdata)
        np.save(os.path.join(save_dir, "ydata.npy"), ydata)  # Always 1-step
        np.save(os.path.join(save_dir, f"ydata_k{k_iterate}.npy"), ydata_k)  # k-step
        np.save(os.path.join(save_dir, "grid.npy"), grid4d)
        np.save(os.path.join(save_dir, "basis_spec.npy"), np.array(B, dtype=object))

        # Save metadata
        metadata = {
            "nf": nf,
            "npix": npix,
            "mx": mx,
            "k_iterate": k_iterate,
            "a": a,
            "b": b,
            "delta": delta,
            "seed": seed,
        }
        np.save(os.path.join(save_dir, "metadata.npy"), metadata)

        print(f"Saved data to {save_dir}")
        print(f"  - xdata.npy: input functions")
        print(f"  - ydata.npy: 1-step transfer operator (for operator loss)")
        print(
            f"  - ydata_k{k_iterate}.npy: {k_iterate}-step transfer operator (for output AE loss)"
        )

    return B, xdata, ydata, ydata_k, pts


# ==========================================
# Plotting
# ==========================================


def quick_plot(xdata, ydata, ydata_k, k_iterate, pts, outfile):
    """Generate diagnostic plot with colorbars."""
    n_samples = min(4, xdata.shape[0])
    n_rows = 3 if k_iterate > 1 else 2
    fig, ax = plt.subplots(n_rows, n_samples, figsize=(4.5 * n_samples, 4 * n_rows))

    if n_samples == 1:
        ax = ax[:, None]

    for i in range(n_samples):
        im0 = ax[0, i].imshow(xdata[i], origin="lower", cmap="Blues", aspect="equal")
        ax[0, i].set_title(f"Input $f_{{{i}}}$")
        fig.colorbar(im0, ax=ax[0, i], fraction=0.046, pad=0.04)

        im1 = ax[1, i].imshow(ydata[i], origin="lower", cmap="Blues", aspect="equal")
        ax[1, i].set_title(f"$\\mathcal{{L}}(f_{{{i}}})$")
        fig.colorbar(im1, ax=ax[1, i], fraction=0.046, pad=0.04)

        if k_iterate > 1:
            im2 = ax[2, i].imshow(
                ydata_k[i], origin="lower", cmap="Blues", aspect="equal"
            )
            ax[2, i].set_title(f"$\\mathcal{{L}}^{{{k_iterate}}}(f_{{{i}}})$")
            fig.colorbar(im2, ax=ax[2, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved diagnostic plot to {outfile}")


def main():
    p = argparse.ArgumentParser(
        description="Generate transfer operator training data (JAX)"
    )
    p.add_argument("--n_functions", type=int, required=True, help="Number of functions")
    p.add_argument("--n_points", type=int, required=True, help="Grid resolution")
    p.add_argument("--max_order", type=int, required=True, help="Max Fourier order")
    p.add_argument("--saving_directory", required=True, help="Output directory")
    p.add_argument(
        "--k_iterate",
        type=int,
        default=1,
        help="Number of operator iterations for output loss",
    )
    p.add_argument("--delta", type=float, default=0.01, help="Cat map perturbation")
    p.add_argument("--a", type=float, default=0.1, help="Conjugacy param a")
    p.add_argument("--b", type=float, default=0.1, help="Conjugacy param b")
    args = p.parse_args()

    base = os.path.join(
        args.saving_directory,
        f"Conjugated_TO_JAX-{args.n_functions}fs-{args.n_points}ps-{args.max_order}or-k{args.k_iterate}",
    )

    print("Building datasets with JAX-accelerated conjugated map...")
    B, xdata, ydata, ydata_k, pts = build_datasets(
        args.n_functions,
        args.n_points,
        args.max_order,
        k_iterate=args.k_iterate,
        a=args.a,
        b=args.b,
        delta=args.delta,
        save_dir=base,
        seed=42,
    )

    print("Generating diagnostic plot...")
    quick_plot(
        xdata,
        ydata,
        ydata_k,
        args.k_iterate,
        pts,
        os.path.join(base, "data_generated.png"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
