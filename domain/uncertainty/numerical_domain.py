"""Shared analytical and sampling covariance-domain checks."""
import numpy as np

def validate_covariance_matrix(mean, covariance_matrix):
    """Allow only scale-relative floating-point roundoff in symmetry and PSD."""
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(covariance_matrix, dtype=np.float64)

    def refuse(reason: str) -> None:
        raise ValueError(reason)

    if mean.ndim != 1:
        refuse("The covariance mean vector must be one-dimensional.")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        refuse("The covariance matrix must be square and two-dimensional.")
    if cov.shape[0] != mean.size:
        refuse("The covariance dimension does not match its sampled vector.")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        refuse("The covariance block contains NaN or infinity.")
    if cov.size == 0:
        return
    scale = max(float(np.max(np.abs(cov))), np.finfo(np.float64).tiny)
    symmetry_atol = 10.0 * np.finfo(np.float64).eps * scale
    if not np.allclose(cov, cov.T, rtol=1.0e-10, atol=symmetry_atol):
        refuse("The covariance matrix is not symmetric within numerical tolerance.")
    if np.any(np.diag(cov) < 0.0):
        refuse("The covariance matrix has a negative diagonal entry.")
    try:
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(cov)))
    except np.linalg.LinAlgError as exc:
        raise ValueError("The covariance eigenvalues could not be evaluated.") from exc
    psd_tolerance = 100.0 * np.finfo(np.float64).eps * max(cov.shape[0], 1) * scale
    if min_eigenvalue < -psd_tolerance:
        refuse(
            "The covariance matrix is not positive semidefinite within numerical tolerance."
        )

