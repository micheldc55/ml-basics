import pytest
import numpy as np

from ml_core.linear_models.linear_regression import LinearRegression, RidgeRegression


_ROUND_DIGITS_TESTS_ = 4


@pytest.fixture
def regression_model():
    return LinearRegression()

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 2) 
    y = 3 + 2 * X[:, 0] + 4 * X[:, 1] + np.random.randn(100) * 0.5 
    return X, y

@pytest.fixture
def known_data():
    X = np.array([[1], [2], [3]])
    y = np.array([5, 7, 9])  # y = 3 + 2x
    return X, y

@pytest.fixture
def trained_linear_regression():
    lr = LinearRegression()
    lr.coef_ = np.array([2, 4])
    lr.intercept_ = 3
    lr.betas_ = np.array([3, 2, 4])
    lr.degrees_of_freedom = 98
    lr.num_predictors = 2
    lr.trained = True
    return lr

@pytest.fixture
def trained_model_on_sample_data(sample_data):
    X, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    return model

@pytest.fixture
def trained_model_on_known_data(known_data):
    X, y = known_data
    model = LinearRegression()
    model.fit(X, y)
    return model

@pytest.fixture
def trained_ridge_regression(known_data):
    X, y = known_data
    ridge = RidgeRegression(lambda_=1.0)
    ridge.fit(X, y)
    return ridge

@pytest.fixture
def trained_null_ridge_on_known_data(known_data):
    X, y = known_data
    ridge = RidgeRegression(lambda_=0.0)
    ridge.fit(X, y)
    return ridge


def test_fit_model(trained_model_on_sample_data):
    """Test if the model fits without errors and sets the trained flag."""
    assert trained_model_on_sample_data.trained, "Model should be marked as trained after fitting"
    assert trained_model_on_sample_data.coef_ is not None, "Model coefficients should be defined after fitting"
    assert trained_model_on_sample_data.intercept_ is not None, "Model intercept should be defined after fitting"

def test_predict_format(trained_model_on_sample_data, sample_data):
    """Test if predictions have the correct shape."""
    X, y = sample_data
    y_pred = trained_model_on_sample_data.predict(X)
    assert y_pred.shape == y.shape, "Prediction output should have the same shape as the input target data"

def test_predict_values(trained_linear_regression):
    """Test if predictions match expected values on known data."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y_pred = trained_linear_regression.predict(X)
    np.testing.assert_array_almost_equal(y_pred, [13, 25, 37], decimal=1)

def test_get_params(trained_model_on_known_data):
    """Test if the model returns the correct parameters."""
    params = trained_model_on_known_data.get_params()
    params = {k: round(v, _ROUND_DIGITS_TESTS_) if isinstance(v, float) else v for k, v in params.items()}
    expected_params = {
        "intercept": 3, 
        "beta_x1": 2.0, 
        "degrees_of_freedom": 1.0,
        "num_predictors": 1.0
    }
    assert params == expected_params, "Model parameters should match the expected values"

def test_compute_r_squared(trained_model_on_known_data):
    """Compute and compare R-squared on known data."""
    r_squared = trained_model_on_known_data.compute_r_squared()
    assert round(r_squared, _ROUND_DIGITS_TESTS_) == 1.0, "R-squared should be 1.0 for a perfect fit"

def test_compute_adjusted_r_squared(trained_model_on_known_data):
    """Compute and compare adjusted R-squared on known data."""
    adjusted_r_squared = trained_model_on_known_data.compute_adjusted_r_squared()
    assert round(adjusted_r_squared, _ROUND_DIGITS_TESTS_) == 1.0, "Adjusted R-squared should be 1.0 for a perfect fit"

def test_compute_standard_errors_shape(trained_model_on_sample_data):
    """Test if standard errors have the correct shape."""
    standard_errors = trained_model_on_sample_data.compute_standard_errors()
    assert standard_errors.shape == (3,), "Standard errors should be a vector with the same length as the number of predictors"

def test_compute_standard_errors(trained_model_on_sample_data):
    """Test if standard errors are computed correctly."""
    standard_errors = trained_model_on_sample_data.compute_standard_errors()
    expected_standard_errors = np.array([0.1, 0.2, 0.2])
    np.testing.assert_array_almost_equal(standard_errors, expected_standard_errors, decimal=1)

def test_compute_t_statistics(trained_model_on_sample_data):
    """Test if t-statistics are computed correctly."""
    t_statistics = trained_model_on_sample_data.compute_t_statistics()
    expected_t_statistics = np.array([22.2, 13.1, 23.6])
    np.testing.assert_array_almost_equal(t_statistics, expected_t_statistics, decimal=1)

def test_compute_f_statistic(trained_model_on_sample_data):
    """Test if F-statistic is computed correctly."""
    f_statistic = trained_model_on_sample_data.compute_f_statistic()
    assert round(f_statistic, 1) == 352.5, "F-statistic should be 352.5"

def test_get_residuals_shape(trained_model_on_sample_data):
    """Test if residuals have the correct shape."""
    residuals = trained_model_on_sample_data.get_residuals()
    assert residuals.shape == (100,), "Residuals should have the same shape as the target data"

def test_get_residuals(trained_model_on_known_data):
    """Test if residuals are computed correctly."""
    residuals = trained_model_on_known_data.get_residuals()
    expected_residuals = np.array([0, 0, 0])
    np.testing.assert_array_almost_equal(residuals, expected_residuals, decimal=_ROUND_DIGITS_TESTS_)

def test_get_residuals_summary_stats_format(trained_model_on_sample_data):
    """Test if residuals summary stats have the correct format."""
    residuals_summary_stats = trained_model_on_sample_data.get_residuals_summary_stats()
    assert isinstance(residuals_summary_stats, dict), "Residuals summary stats should be a dictionary"
    expected_keys = {'min', '1Q', 'median', '3Q', 'max'}
    assert set(residuals_summary_stats.keys()) == expected_keys, "Residuals summary stats should have the correct keys"

def test_get_residual_standard_error(trained_model_on_known_data):
    """Test if residual standard error is computed correctly."""
    residual_standard_error = trained_model_on_known_data.get_residual_standard_error()
    assert round(residual_standard_error, _ROUND_DIGITS_TESTS_) == 0.0, "Residual standard error should be 0.0 for a perfect fit"

def test_fit_non_invertible_matrix(regression_model):
    """Test if model raises ValueError when matrix is not invertible."""
    X = np.array([[1, 2], [2, 4]])
    y = np.array([1, 2])
    with pytest.raises(ValueError):
        regression_model.fit(X, y)

def test_degrees_of_freedom_error(regression_model):
    """Test if model raises ValueError when degrees of freedom <= 0."""
    X = np.array([[1, 2, 3]])
    y = np.array([1, 1])
    with pytest.raises(ValueError):
        regression_model.fit(X, y)

def test_repr_str(regression_model, sample_data):
    """Test if __repr__ and __str__ return strings."""
    X, y = sample_data
    regression_model.fit(X, y)
    model_repr = repr(regression_model)
    model_str = str(regression_model)
    assert isinstance(model_repr, str), "__repr__ should return a string"
    assert isinstance(model_str, str), "__str__ should return a string"

def test_ridge_regression_fit_attributes(trained_ridge_regression):
    """Test if Ridge regression fits without errors and sets the trained flag."""
    assert trained_ridge_regression.trained, "Model should be marked as trained after fitting"
    assert trained_ridge_regression.coef_ is not None, "Model coefficients should be defined after fitting"
    assert trained_ridge_regression.intercept_ is not None, "Model intercept should be defined after fitting"

def test_null_ridge_equals_simple_regression(trained_null_ridge_on_known_data, trained_model_on_known_data):
    """Test if Ridge regression with alpha=0.0 equals simple linear regression."""
    np.testing.assert_array_almost_equal(
        trained_null_ridge_on_known_data.betas_, trained_model_on_known_data.betas_, decimal=_ROUND_DIGITS_TESTS_
    )

def test_ridge_predict(trained_ridge_regression, known_data):
    """Test if Ridge regression predictions match expected values on known data."""
    X, _ = known_data
    y_pred = trained_ridge_regression.predict(X)
    assert len(y_pred) == 3, "Ridge regression predictions should have the same length as the input data"

def test_ridge_coefficients_shrinkage(sample_data):
    """Test that coefficients shrink as lambda increases."""
    X, y = sample_data
    lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    norms = []
    for lambda_ in lambdas:
        model = RidgeRegression(lambda_=lambda_)
        model.fit(X, y)
        coef_norm = np.linalg.norm(model.coef_)
        norms.append(coef_norm)
    
    assert norms == sorted(norms, reverse=True), "Coefficients should shrink as lambda increases"

def test_negative_lambda_ridge(known_data):
    """Test if Ridge regression raises ValueError when lambda is negative."""
    X, y = known_data
    with pytest.raises(ValueError):
        RidgeRegression(lambda_=-1.0).fit(X, y)