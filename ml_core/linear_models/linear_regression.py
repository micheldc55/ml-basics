import numpy as np
from numpy.typing import ArrayLike

from ml_core.core_components.model import Model
from ml_core.core_components.linalg import is_invertible_svd


class LinearRegression(Model):
    """Linear regression model."""
    def __init__(self):
        super().__init__(name="LinearRegression")
        self.coef_ = None
        self.intercept_ = None
        self.betas_ = None

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the model to the data.
        
        Args:
            X (ArrayLike): The input data.
            y (ArrayLike): The target data.
        """
        X = self._preprocess_input_matrix(X)
        y = np.array(y)

        self._check_is_matrix_invertible(X.T @ X)

        self.betas_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.betas_[0]
        self.coef_ = self.betas_[1:]

    def predict(self, X: ArrayLike):
        """Make predictions using the model."""
        X = np.array(X)
        X = self._include_intercept_column(X)
        return X @ self.betas_
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError
    
    def get_params(self):
        return {"coef_": self.coef_, "intercept_": self.intercept_}
    
    @staticmethod
    def _check_is_matrix_invertible(X: ArrayLike):
        """Checks if a matrix (X) is invertible."""
        if not is_invertible_svd(X):
            raise ValueError(f"Matrix is not invertible.")
        
    @staticmethod
    def _include_intercept_column(X: ArrayLike):
        """Include an intercept column in the matrix. Adds a column 
        of ones to the matrix X to account for beta_0."""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _preprocess_input_matrix(self, X: ArrayLike) -> ArrayLike:
        """Preprocess the input matrix."""
        x_copy = X.copy()
        x_copy = np.array(x_copy)

        if x_copy.ndim == 1:
            x_copy = x_copy.reshape(-1, 1)

        return self._include_intercept_column(x_copy)
    
    def __repr__(self):
        return "\n".join([f"beta_{i}: {beta}" for i, beta in enumerate(self.betas_)])

    def __str__(self):
        return self.__repr__()