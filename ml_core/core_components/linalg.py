import numpy as np
from numpy.typing import ArrayLike


def invert_matrix(X: ArrayLike):
    """Compute the inverse of a matrix."""
    return np.linalg.inv(X)


def is_symmetric(X: ArrayLike) -> bool:
    """Check if a matrix is symmetric."""
    return np.allclose(X, X.T)


def is_invertible_np(X: ArrayLike) -> bool:
    """Check if a matrix is invertible."""
    try:
        invert_matrix(X)
        return True
    except np.linalg.LinAlgError:
        return False
    

def is_invertible_svd(X: ArrayLike, epsilon: float = 1e-10) -> bool:
    """Check if a matrix is invertible using SVD.
    
    Args:
        X (ArrayLike): The matrix to check.
        epsilon (float): The threshold for the singular values. 


    Returns:
        bool: True if the matrix is invertible, False otherwise.
    """
    U, S, V = np.linalg.svd(X)
    return np.all(S > epsilon)