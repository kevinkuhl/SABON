import numpy as np
from scipy import linalg


def galerkin_Lphi(
    basis_matrices,
    operator_action,
    inner_product_weight=None,
    normalize_eigenvectors=True,
    return_left_vectors=False,
):
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
        mv = np.clip(mv, 0, None)
        eps = 1e-14
        Rvecs /= np.sqrt(mv + eps)[None, :]
        if return_left_vectors:
            Lvecs /= (mv + eps)[None, :]

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
        mv = np.clip(mv, 0, None)
        eps = 1e-14
        R /= np.sqrt(mv + eps)[None, :]
        if return_left_vectors:
            L /= (mv + eps)[None, :]

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
