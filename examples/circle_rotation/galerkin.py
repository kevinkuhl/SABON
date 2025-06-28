import numpy as np
from scipy import linalg


def galerkin_Lphi(
    basis_matrices,
    operator_action,
    inner_product_weight=None,
    normalize_eigenvectors=True,
    return_left_vectors=False,
):
    """
    Parameters
    ----------
    basis_matrices      : sequence of arrays, φ_i(x) on the grid
    operator_action     : sequence of arrays, Lφ_i(x) (same shapes as basis_matrices)
    inner_product_weight: quadrature weights w(x) or None (defaults to ones)
    normalize_eigenvectors : if True, scale right eigenvectors so vᴴ M v = 1
    return_left_vectors : if True, also return the left eigenvectors

    Returns
    -------
    dict with
        eigenvalues
        eigenvectors
        eigenfunctions
        A, M
        left_eigenvectors
    """
    if len(operator_action) != len(basis_matrices):
        raise ValueError("operator_action must have the same length as basis_matrices.")

    Ny, Nx = basis_matrices[0].shape
    n_pts = Ny * Nx

    B = np.stack([b.ravel() for b in basis_matrices], axis=1)
    Lphi = np.stack([f.ravel() for f in operator_action], axis=1)

    w = (
        np.ones(n_pts, float)
        if inner_product_weight is None
        else inner_product_weight.ravel().astype(float)
    )

    M = B.conj().T @ (w[:, None] * B)
    A = B.conj().T @ (w[:, None] * Lphi)

    if return_left_vectors:
        eigvals, Lvecs, Rvecs = linalg.eig(A, M, left=True, right=True)
    else:
        eigvals, Rvecs = linalg.eig(A, M)

    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    Rvecs = Rvecs[:, order]
    if return_left_vectors:
        Lvecs = Lvecs[:, order]

    if normalize_eigenvectors:
        mv = np.einsum("bi,ij,bi->b", Rvecs.conj().T, M, Rvecs).real
        tiny = mv < 1e-12
        if tiny.any():
            print("zero / tiny vᴴMv for modes:", np.where(tiny)[0])
            mv[tiny] = 1.0
        Rvecs /= np.sqrt(mv)[None, :]
        if return_left_vectors:
            Lvecs /= mv[None, :]

    basis_stack = np.stack(basis_matrices, axis=0)
    eigfuncs = np.tensordot(Rvecs.T, basis_stack, axes=(1, 0))

    out = dict(
        eigenvalues=eigvals,
        eigenvectors=Rvecs,
        eigenfunctions=eigfuncs,
        A=A,
        M=M,
    )
    if return_left_vectors:
        out["left_eigenvectors"] = Lvecs
    return out


def galerkin_phiT(
    basis_matrices,
    transformed_basis,
    inner_product_weight=None,
    normalize_eigenvectors=True,
    return_left_vectors=False,
):
    """
    Returns a dict with
        eigenvalues
        eigenvectors
        eigenfunctions
        A, M
        left_eigenvectors
    """
    Ny, Nx = basis_matrices[0].shape
    n_pts = Ny * Nx

    B = np.stack([b.ravel() for b in basis_matrices], axis=1)
    BT = np.stack([b.ravel() for b in transformed_basis], axis=1)

    if inner_product_weight is None:
        w = np.ones(n_pts, float)
    else:
        w = inner_product_weight.ravel().astype(float)

    M = B.conj().T @ (w[:, None] * B)
    A = BT.conj().T @ (w[:, None] * B)

    if return_left_vectors:
        eigvals, L, R = linalg.eig(A, M, left=True, right=True)
    else:
        eigvals, R = linalg.eig(A, M)

    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    R = R[:, order]
    if return_left_vectors:
        L = L[:, order]

    if normalize_eigenvectors:
        mv = np.einsum("bi,ij,bi->b", R.conj().T, M, R).real
        R /= np.sqrt(mv)[None, :]
        if return_left_vectors:
            L /= mv[None, :]

    basis_stack = np.stack(basis_matrices, axis=0)
    eigfuncs = np.tensordot(R.T, basis_stack, axes=(1, 0))

    out = dict(
        eigenvalues=eigvals,
        eigenvectors=R,
        eigenfunctions=eigfuncs,
        A=A,
        M=M,
    )
    if return_left_vectors:
        out["left_eigenvectors"] = L
    return out
