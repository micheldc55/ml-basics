import numpy as np
from numpy.typing import ArrayLike

from ml_core.core_components.model import Model
from ml_core.core_components.linalg import is_invertible_svd


class LinearRegression(Model):
    """Linear regression model."""
    def __init__(self, compute_stats: bool = False):
        super().__init__(name="LinearRegression")
        self.coef_ = None
        self.intercept_ = None
        self.betas_ = None
        self.degrees_of_freedom = None
        self.compute_stats = compute_stats

    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the model to the data.
        
        Args:
            X (ArrayLike): The input data.
            y (ArrayLike): The target data.
        """
        X = self._preprocess_input_matrix(X)
        y = self._preprocess_input_vector(y)

        self._check_degrees_of_freedom(X.shape[0] - X.shape[1])
        self.degrees_of_freedom = X.shape[0] - X.shape[1]

        self._check_is_matrix_invertible(X.T @ X)

        print("Shape of X: ", X.shape)
        print("Shape of X.T: ", X.T.shape)
        print("Shape of y: ", y.shape)

        self.betas_ = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = self.betas_[0]
        self.coef_ = self.betas_[1:]

    def predict(self, X: ArrayLike):
        """Make predictions using the model."""
        x_array = self._preprocess_input_mat_if_needed(X)
        return x_array @ self.betas_
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError
    
    def get_params(self):
        return {"coef_": self.coef_, "intercept_": self.intercept_}
    
    def compute_t_statistics(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Compute the t-statistics of the model."""
        x_array = self._preprocess_input_mat_if_needed(X)
        y_array = self._preprocess_input_vector(y)
        standard_errors = self._compute_standard_errors(x_array, y_array)
        return self.betas_ / standard_errors
    
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
    
    @staticmethod
    def _check_degrees_of_freedom(degrees_of_freedom: int):
        """Check if the degrees of freedom are valid."""
        if degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be greater than 0.")
    
    def _preprocess_input_matrix(self, X: ArrayLike) -> ArrayLike:
        """Preprocess the input matrix."""
        x_copy = X.copy()
        x_copy = np.array(x_copy)

        if x_copy.ndim == 1:
            x_copy = x_copy.reshape(-1, 1)

        return self._include_intercept_column(x_copy)
    
    def _check_preprocess_is_needed(self, X: ArrayLike) -> bool:
        """Check if the input matrix needs to be preprocessed."""
        beta_len = len(self.betas_) if self.betas_ is not None else 0
        if X.shape[1] != beta_len:
            return True
        return False
    
    def _preprocess_input_mat_if_needed(self, X: ArrayLike) -> ArrayLike:
        """Preprocess the input matrix if needed."""
        if self._check_preprocess_is_needed(X):
            return self._preprocess_input_matrix(X)
        return X
    
    def _preprocess_input_vector(self, y: ArrayLike) -> ArrayLike:
        """Preprocess the input vector."""
        return np.array(y).copy()
    
    def _get_residuals(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Get the residuals of the model."""
        y_vec = self._preprocess_input_vector(y)
        return y_vec - self.predict(X)
    
    def _get_residuals_variance(self, X: ArrayLike, y: ArrayLike) -> float:
        """Get the variance of the residuals."""
        residuals = self._get_residuals(X, y)
        return np.sum(np.square(residuals)) / self.degrees_of_freedom
    
    def _compute_covariance_matrix(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Compute the covariance matrix of the model."""
        residuals_variance = self._get_residuals_variance(X, y)
        return residuals_variance * np.linalg.inv(X.T @ X)
    
    def _compute_standard_errors(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Compute the standard errors of the model."""
        covariance_matrix = self._compute_covariance_matrix(X, y)
        return np.sqrt(np.diag(covariance_matrix))
    
    def __repr__(self):
        return "\n".join([f"beta_{i}: {beta}" for i, beta in enumerate(self.betas_)])

    def __str__(self):
        return self.__repr__()