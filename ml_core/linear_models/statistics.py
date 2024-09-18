import scipy

import numpy as np
from numpy.typing import ArrayLike

from ml_core.linear_models.linear_regression import LinearRegression


class TTest:
    """Implements a simple t-test class that computes the p-value for a given 
    dataset."""
    def __init__(self, t_statistics: ArrayLike, degrees_of_freedom: int):
        self.t_statistics = t_statistics
        self.degrees_of_freedom = degrees_of_freedom
        self.p_values = self._compute_p_values()

    def _compute_p_values(self) -> ArrayLike:
        prob_T_higher_than_t = 1 - scipy.stats.t.cdf(
            np.abs(self.t_statistics), self.degrees_of_freedom
        )
        return 2 * prob_T_higher_than_t


class FTest:
    pass


if __name__ == "__main__":
    X = np.array([[1.01, 0], [2.0004, 0], [3, 0], [4, 0.1], [5, 0.01]])
    y = np.array([2, 3, 4, 5, 6])

    model = LinearRegression()
    model.fit(X, y)
    t_statistics = model.compute_t_statistics(X, y)
    t_test = TTest(t_statistics, model.degrees_of_freedom)
    print(t_test.p_values)
