import pytest
import numpy as np

from ml_core.linear_models.linear_regression import LinearRegression

@pytest.fixture
def regression_model():
    return LinearRegression()

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 2) 
    y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) * 0.5 
    return X, y

def test_fit_model(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    assert regression_model.trained, "Model should be marked as trained after fitting"
    assert regression_model.coef_ is not None, "Model coefficients should be defined after fitting"
    assert regression_model.intercept_ is not None, "Model intercept should be defined after fitting"

def test_predict(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    y_pred = regression_model.predict(X)
    assert y_pred.shape == y.shape, "Prediction output should have the same shape as the input target data"

def test_r_squared(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    r_squared = regression_model.compute_r_squared()
    assert 0 <= r_squared <= 1, "R-squared value should be between 0 and 1"

def test_adjusted_r_squared(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    adjusted_r_squared = regression_model.compute_adjusted_r_squared()
    assert 0 <= adjusted_r_squared <= 1, "Adjusted R-squared should be between 0 and 1"

def test_residuals_summary_stats(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    residuals_summary = regression_model.get_residuals_summary_stats()
    assert "min" in residuals_summary, "Summary should include minimum residual"
    assert "max" in residuals_summary, "Summary should include maximum residual"
    assert "median" in residuals_summary, "Summary should include median residual"
    assert "1Q" in residuals_summary, "Summary should include first quartile"
    assert "3Q" in residuals_summary, "Summary should include third quartile"

def test_f_statistic(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    f_statistic = regression_model.compute_f_statistic()
    assert f_statistic >= 0, "F-statistic should be non-negative"

def test_t_statistics(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    t_statistics = regression_model.compute_t_statistics()
    assert t_statistics.shape == regression_model.betas_.shape, "T-statistics should match the shape of the model coefficients"

def test_residual_standard_error(regression_model, sample_data):
    X, y = sample_data
    regression_model.fit(X, y)
    residual_standard_error = regression_model.get_residual_standard_error()
    assert residual_standard_error > 0, "Residual standard error should be positive"

def test_invalid_matrix_inversion(regression_model):
    X = np.ones((10, 2)) 
    y = np.random.rand(10)
    with pytest.raises(ValueError, match="Matrix is not invertible."):
        regression_model.fit(X, y)

def test_degrees_of_freedom(regression_model, sample_data):
    X, y = sample_data
    with pytest.raises(ValueError, match="Degrees of freedom must be greater than 0."):
        regression_model._check_degrees_of_freedom(-1)
