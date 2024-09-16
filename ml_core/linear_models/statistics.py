import scipy

import numpy as np
from numpy.typing import ArrayLike

from ml_core.linear_models.linear_regression import LinearRegression


class LRStatistics:
    def __init__(self, model: LinearRegression, X: ArrayLike, y: ArrayLike):
        self.model = model
        self.data = X
        self.target = y
        self.X = self.model._preprocess_input_matrix(X)
        self.y = self.model._preprocess_input_vector(y)

        self.beta_p_vals = self._compute_beta_p_values()


    def _get_residuals(self) -> ArrayLike:
        return self.model.get_residuals(self.data, self.target)
    
    def _get_sum_squared_residuals(self) -> float:
        residuals = self.get_residuals()
        return np.sum(np.square(residuals))
    
    def _get_residual_variance(self) -> float:
        return self._get_sum_squared_residuals() / (self.X.shape[0] - self.X.shape[1] - 1)
    
    def _get_beta_covariance_matrix(self) -> ArrayLike:
        residual_variance = self._get_residual_variance()
        return residual_variance * np.linalg.inv(self.X.T @ self.X)
    
    def _get_beta_std_errs(self) -> ArrayLike:
        return np.sqrt(np.diag(self._get_beta_covariance_matrix()))
    
    def _compute_beta_t_test_statistic(self) -> ArrayLike:
        beta_std_errs = self._get_beta_std_errs()
        return self.model.betas_ / beta_std_errs
    
    def _compute_beta_p_values(self) -> ArrayLike:
        t_stats = self._compute_beta_t_test_statistic()
        degrees_of_freedom = self.X.shape[0] - self.X.shape[1] - 1
        prob_T_higher_than_t = 1 - scipy.stats.t.cdf(np.abs(t_stats), degrees_of_freedom)
        return 2 * prob_T_higher_than_t
    
    def _get_sum_total_squares(self) -> float:
        return np.sum(np.square(self.y - np.mean(self.y)))
    
    