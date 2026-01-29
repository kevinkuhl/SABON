import numpy as np
import torch
from scipy import linalg


def eigs(
    model,
    t_in,
    normalize_eigenvectors: bool = True,
    return_left_vectors: bool = False,
):
    model.eval()
    with torch.no_grad():
        t_tensor = torch.as_tensor(
            t_in,
            dtype=torch.get_default_dtype(),
            device=next(model.parameters()).device,
        )
        B = model.Encoder(t_tensor).cpu().numpy().T

    try:
        w_flat = model.trap_w_flat.to(torch.float64).cpu().numpy()
    except AttributeError as e:
        raise AttributeError(
            "The model must expose a tensor 'trap_w_flat' with quadrature weights."
        ) from e

    M = (B * w_flat) @ B.T

    G_layer = model.G.layers[0]

    G = G_layer.weight.detach().cpu().numpy()

    A = G @ M

    if return_left_vectors:
        eigvals, Lvecs, Rvecs = linalg.eig(A, left=True, right=True)
    else:
        eigvals, Rvecs = linalg.eig(A)

    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    Rvecs = Rvecs[:, order]
    if return_left_vectors:
        Lvecs = Lvecs[:, order]

    if normalize_eigenvectors:
        mv = np.einsum("bi,ij,bi->b", Rvecs.conj().T, M, Rvecs).real
        eps = 1e-14
        Rvecs /= np.sqrt(np.maximum(mv, 0) + eps)[None, :]
        if return_left_vectors:
            Lvecs /= (mv + eps)[None, :]

    eigfuncs = Rvecs.T @ B

    out = {
        "eigenvalues": eigvals,
        "eigenvectors": Rvecs,
        "eigenfunctions": eigfuncs,
        "A": A,
        "M": M,
    }
    if return_left_vectors:
        out["left_eigenvectors"] = Lvecs

    return out
