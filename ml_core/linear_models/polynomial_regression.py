"""Polynomial regression is just a linear regression model with polynomial features.

Given X = [[1, 2], [2, 3], [3, 4]] and y = [2, 3, 4], we can transform X to include polynomial features
of degree 2 as follows:

X = [[1, 2, 1, 4, 2, 4], [2, 3, 4, 9, 6, 12], [3, 4, 9, 16, 12, 16]]

For row i in X, the polynomial features are [1, x_i1, x_i1^2, x_i2, x_i1*x_i2, x_i2^2].
"""
import numpy as np
from itertools import combinations_with_replacement
from numpy.typing import ArrayLike

from ml_core.linear_models.linear_regression import LinearRegression


class PolynomialTransformer:
    def __init__(self, degree: int = 2):
        self._check_degree(degree)
        self.degree = degree


    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform the input matrix to include polynomial features."""
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        comb_indices = self._get_column_index_combinations(n_features)
        poly_features = self._construct_polynomial_features(X, comb_indices, n_samples)

        return poly_features
    

    def _get_column_index_combinations(self, n_features: int):
        """Get the column index combinations for the polynomial features."""
        return list(combinations_with_replacement(range(n_features), self.degree))
    
    
    def _construct_polynomial_features(
            self, 
            X: ArrayLike, 
            comb_indices: ArrayLike,
            n_samples: int
        ) -> ArrayLike:
        """Construct the polynomial features for the input matrix."""
        poly_features = np.ones((n_samples, len(comb_indices)))
        for index, comb in enumerate(comb_indices):
            poly_features[:, index] = np.prod(X[:, comb], axis=1)
        return poly_features

    
    @staticmethod
    def _check_degree(degree: int):
        """Check if the degree is valid."""
        if degree < 1:
            raise ValueError("Degree must be greater than 0.")
        

class PolynomialRegression(LinearRegression):
    """Implements a polynomial regression model (includes feature 
    preprocessing).
    """
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree
        self.poly_transformer = PolynomialTransformer(degree)


    def fit(self, X: ArrayLike, y: ArrayLike):
        """Fit the polynomial regression model."""
        X_poly = self.poly_transformer.transform(X)
        super().fit(X_poly, y)


    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict the target variable."""
        X_poly = self.poly_transformer.transform(X)
        return super().predict(X_poly)
    

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform the input matrix to include polynomial features."""
        return self.poly_transformer.transform(X)