import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi


def cat_map(x, delta=0.01):
    y0 = 2.0 * x[..., 0] + x[..., 1] + 2.0 * delta * np.cos(2 * pi * x[..., 0])
    y1 = x[..., 0] + x[..., 1] + delta * np.sin(4 * pi * x[..., 1] + 1)
    return np.mod(np.stack((y0, y1), -1), 1.0)


def make_1d_basis(m):
    r2 = np.sqrt(2.0)
    b = [("c", 0, 1.0)]
    for k in range(1, m + 1):
        b += [("c", k, r2), ("s", k, r2)]
    return b


def make_2d_basis(m):
    b1 = make_1d_basis(m)
    return [(t1, k1, t2, k2, s1 * s2) for t1, k1, s1 in b1 for t2, k2, s2 in b1]


def eval_basis_2d(B, x1, y1, x2, y2):
    tp = 2 * pi
    t1 = (np.arctan2(y1, x1) % (2 * pi)) / tp
    t2 = (np.arctan2(y2, x2) % (2 * pi)) / tp
    v = np.empty((len(B), t1.size))
    for i, (a, kA, b, kB, s) in enumerate(B):
        fA = np.cos if a == "c" else np.sin
        fB = np.cos if b == "c" else np.sin
        v[i] = s * fA(tp * kA * t1) * fB(tp * kB * t2)
    return v


def random_linear_combinations(B, n, rng):
    nB = len(B)
    C = rng.uniform(-1, 1, (n, nB))

    def mk(c):
        return lambda x1, y1, x2, y2: (c @ eval_basis_2d(B, x1, y1, x2, y2)).ravel()

    return [mk(c) for c in C]


def midpoint_grid(n):
    p = (np.arange(n) + 0.5) / n
    th = 2 * pi * p
    return p, np.cos(th), np.sin(th)


def inv_cat_map_grid(n, delta):
    p, _x, _y = midpoint_grid(n)
    g = np.array([[px, py] for py in p for px in p])

    def det(X):
        x1, x2 = X
        return (2 - 4 * pi * delta * np.sin(2 * pi * x1)) * (
            1 + 4 * pi * delta * np.cos(4 * pi * x2 + 1)
        ) - 1

    def Tinv(y):
        x = np.dot([[1, -1], [-1, 2]], y) % 1.0
        for _ in range(30):
            r = (y - cat_map(x, delta) + 0.5) % 1.0 - 0.5
            if np.linalg.norm(r) < 1e-12:
                break
            x1, x2 = x
            J = [
                [2 - 4 * pi * delta * np.sin(2 * pi * x1), 1],
                [1, 1 + 4 * pi * delta * np.cos(4 * pi * x2 + 1)],
            ]
            x = (x + np.linalg.solve(J, r)) % 1.0
        return x

    inv = np.array([Tinv(p) for p in g])
    d = np.abs([det(p) for p in inv])
    return inv, d


def build_datasets(nf, npix, mx, delta=0.01, seed=42, save_dir=None):
    rng = np.random.default_rng(seed)
    B = make_2d_basis(mx)
    F = random_linear_combinations(B, nf, rng)
    pts, x1, y1 = midpoint_grid(npix)
    _, x2, y2 = midpoint_grid(npix)
    X1, X2 = np.meshgrid(x1, x2, indexing="ij")
    Y1, Y2 = np.meshgrid(y1, y2, indexing="ij")
    grid4d = np.stack([X1, Y1, X2, Y2], -1)
    xf, yf, uf, vf = (grid4d[..., i].ravel() for i in range(4))
    basis_flat = eval_basis_2d(B, xf, yf, uf, vf)
    basis_mats = np.rot90(basis_flat.reshape(len(B), npix, npix), 2, (1, 2))
    inv_pts, det = inv_cat_map_grid(npix, delta)
    t1 = 2 * pi * inv_pts[:, 0]
    t2 = 2 * pi * inv_pts[:, 1]
    inv_x, inv_y = np.cos(t1), np.sin(t1)
    inv_u, inv_v = np.cos(t2), np.sin(t2)
    phiT_flat = eval_basis_2d(B, inv_x, inv_y, inv_u, inv_v) / det
    phiT_mats = np.rot90(phiT_flat.reshape(len(B), npix, npix), 2, (1, 2))
    xdata = np.empty((nf, npix, npix))
    ydata = np.empty_like(xdata)
    for j, f in enumerate(F):
        xdata[j] = f(xf, yf, uf, vf).reshape(npix, npix)
        ydata[j] = (f(inv_x, inv_y, inv_u, inv_v) / det).reshape(npix, npix)
        ydata[j] = np.rot90(ydata[j], 2)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "xdata.npy"), xdata)
        np.save(os.path.join(save_dir, "ydata.npy"), ydata)
        np.save(os.path.join(save_dir, "grid.npy"), grid4d)
        np.save(os.path.join(save_dir, "basis.npy"), basis_mats)
        np.save(os.path.join(save_dir, "basis_phiT.npy"), phiT_mats)
        np.save(os.path.join(save_dir, "basis_spec.npy"), np.array(B, dtype=object))
        print("saved data to", save_dir)
    return B, xdata, ydata, pts


def quick_plot(B, xdata, ydata, pts, outfile):
    selB = (0, 25, -1)
    selD = (0, 25, -1)
    tp = 2 * np.pi
    fig, ax = plt.subplots(
        3, len(selB), figsize=(3.5 * len(selB), 9), constrained_layout=True
    )
    XX, YY = np.meshgrid(pts, pts, indexing="ij")
    for k, i in enumerate(selB):
        t1, k1, t2, k2, s = B[i]
        f1 = np.cos if t1 == "c" else np.sin
        f2 = np.cos if t2 == "c" else np.sin
        im = ax[0, k].imshow(
            np.rot90(s * f1(tp * k1 * XX) * f2(tp * k2 * YY), 2),
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="RdBu_r",
        )
        ax[0, k].set_title(f"Basis {i}")
        fig.colorbar(im, ax=ax[0, k], shrink=0.75)
    for k, i in enumerate(selD):
        im = ax[1, k].imshow(
            xdata[i], origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
        )
        ax[1, k].set_title(f"f {i}")
        fig.colorbar(im, ax=ax[1, k], shrink=0.75)
        im = ax[2, k].imshow(
            ydata[i], origin="lower", extent=[0, 1, 0, 1], cmap="viridis"
        )
        ax[2, k].set_title(f"L f {i}")
        fig.colorbar(im, ax=ax[2, k], shrink=0.75)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_functions", type=int, required=True)
    p.add_argument("--n_points", type=int, required=True)
    p.add_argument("--max_order", type=int, required=True)
    p.add_argument("--saving_directory", required=True)
    args = p.parse_args()
    base = os.path.join(
        args.saving_directory,
        f"TO_perturbed_cat_map_4D-{args.n_functions}fs-{args.n_points}ps-{args.max_order}or",
    )
    print("building datasets...")
    B, xdata, ydata, pts = build_datasets(
        args.n_functions, args.n_points, args.max_order, save_dir=base, seed=42
    )
    print("generating diagnostic plot...")
    quick_plot(B, xdata, ydata, pts, os.path.join(base, "data_generated.png"))
    print("done.")


if __name__ == "__main__":
    main()
