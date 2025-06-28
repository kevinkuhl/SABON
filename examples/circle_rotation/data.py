import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def trig_pol_circle_rotation_2d(
    n_functions, n_points, max_order, alpha, seed, save_dir
):
    np.random.seed(seed)
    sine = [
        lambda x, y, k=k: np.sin(k * np.angle(x + 1j * y))
        for k in range(1, max_order + 1)
    ]
    cosn = [
        lambda x, y, k=k: np.cos(k * np.angle(x + 1j * y)) for k in range(max_order + 1)
    ]
    F = []
    for _ in range(n_functions):
        sc = 2 * np.random.rand(max_order) - 1
        cc = 2 * np.random.rand(max_order + 1) - 1
        F.append(
            lambda x, y, s=sc, c=cc: sum(s[i] * sine[i](x, y) for i in range(max_order))
            + sum(c[i] * cosn[i](x, y) for i in range(max_order + 1))
        )
    th = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x, y = np.cos(th), np.sin(th)
    grid = np.stack((x, y), -1).reshape(n_points, 1, 2)
    ca, sa = np.cos(alpha), np.sin(alpha)

    def L(f):
        return lambda x, y: f(ca * x - sa * y, sa * x + ca * y)

    xdata = np.array([f(grid[:, 0, 0], grid[:, 0, 1]).reshape(n_points, 1) for f in F])
    ydata = np.array(
        [L(f)(grid[:, 0, 0], grid[:, 0, 1]).reshape(n_points, 1) for f in F]
    )
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "xdata.npy"), xdata)
        np.save(os.path.join(save_dir, "ydata.npy"), ydata)
        np.save(os.path.join(save_dir, "grid.npy"), grid)
        print("saved data to", save_dir)
    return xdata, ydata, grid


def diagnostic_plot(xdata, ydata, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    im0 = ax[0].imshow(xdata[0, :, 0][None, :], aspect="auto", cmap="viridis")
    im1 = ax[1].imshow(ydata[0, :, 0][None, :], aspect="auto", cmap="viridis")
    ax[0].set_title("f sample")
    ax[1].set_title("L f sample")
    fig.colorbar(im0, ax=ax[0])
    fig.colorbar(im1, ax=ax[1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_functions", type=int, required=True)
    p.add_argument("--n_points", type=int, required=True)
    p.add_argument("--max_order", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--saving_directory", required=True)
    args = p.parse_args()
    out_dir = os.path.join(
        args.saving_directory,
        f"TO_circle_rotation_2D-{args.n_functions}fs-{args.n_points}ps-{args.max_order}or-{args.alpha}alp",
    )
    print("building datasets...")
    xdata, ydata, grid = trig_pol_circle_rotation_2d(
        args.n_functions, args.n_points, args.max_order, args.alpha, 42, out_dir
    )
    print("creating diagnostic plot...")
    diagnostic_plot(xdata, ydata, os.path.join(out_dir, "data_generated.png"))
    print("done.")


if __name__ == "__main__":
    main()
