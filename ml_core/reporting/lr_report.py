import datetime

from numpy import r_

from ml_core.linear_models.linear_regression import LinearRegression
from ml_core.statistics.statistical_tests import TTest, FTest


def generate_linear_regression_report(
    coefficients,
    std_errors,
    t_values,
    p_values,
    residual_summary,
    r_squared,
    adjusted_r_squared,
    residual_standard_error,
    f_statistic,
    f_p_value,
    df_residual,
    df_model,
    lr_model=None
):
    """
    Generate a linear regression report, formatted similarly to R's regression summary.
    """
    # Check if model is fitted
    if lr_model is not None and not getattr(lr_model, 'trained', False):
        raise ValueError("The Linear Regression model has not been trained yet. Fit the model before generating a report!")

    # Printing Call
    print("Call:")
    print("lm(formula = y ~ x1 + x2, data = dataset)")
    print()

    # Printing Residuals
    print("Residuals:")
    print(
        "{:<8} {:<8} {:<8} {:<8} {:<8}".format("Min", "1Q", "Median", "3Q", "Max")
    )
    print(
        "{:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f} {:<8.4f}".format(
            residual_summary['min'],
            residual_summary['1Q'],
            residual_summary['median'],
            residual_summary['3Q'],
            residual_summary['max'],
        )
    )
    print()

    # Printing Coefficients
    print("Coefficients:")
    header = f"{'':>12} {'Estimate':>10} {'Std. Error':>12} {'t value':>8} {'Pr(>|t|)':>10}"
    print(header)
    print("-" * len(header))
    for name, coef, std_err, t_val, p_val in zip(
        coefficients.keys(), coefficients.values(), std_errors, t_values, p_values
    ):
        signif = ""
        if p_val < 0.001:
            signif = "***"
        elif p_val < 0.01:
            signif = "**"
        elif p_val < 0.05:
            signif = "*"
        elif p_val < 0.1:
            signif = "."

        print(
            f"{name:>12} {coef:>10.5f} {std_err:>12.5f} {t_val:>8.3f} {p_val:>10.4g} {signif}"
        )
    print("-" * len(header))
    print("Signif. codes:  0 `***` 0.001 `**` 0.01 `*` 0.05 `.` 0.1 ` ` 1")
    print("-" * len(header))
    print()

    # Printing Residual Standard Error
    print(
        f"Residual std error: \t{residual_standard_error:.3f} on {df_residual} degrees of freedom"
    )

    # Printing R-squared and Adjusted R-squared
    print(f"Multiple R-squared:  \t{r_squared:.3f}")
    print(f"Adjusted R-squared:  \t{adjusted_r_squared:.3f}")

    # Printing F-statistic
    print(
        f"F-statistic: \t{f_statistic:.1f} on {df_model} and {df_residual} DF,  p-value: {f_p_value:.2e}"
    )


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    n_samples = 100
    X = np.random.normal(0, 1, (n_samples, 2))
    epsilon = np.random.normal(0, 0.1, n_samples)

    # Known coefficients
    beta_0 = 4
    beta_1 = 3
    beta_2 = -2

    # Generate target variable
    y = beta_0 + beta_1 * X[:, 0] + beta_2 * X[:, 1] + epsilon

    model = LinearRegression()
    model.fit(X, y)

    coefficients = model.get_params()
    std_errors = model.compute_standard_errors()
    t_values = model.compute_t_statistics()
    t_test = TTest(t_values, model.degrees_of_freedom)
    t_test_p_values = t_test.p_values

    residual_summary = model.get_residuals_summary_stats()
    r_squared = model.compute_r_squared()
    adjusted_r_squared = model.compute_adjusted_r_squared()

    residual_standard_error = model.get_residual_standard_error()
    f_statistic = model.compute_f_statistic()
    f_test = FTest(f_statistic, model.num_predictors, model.degrees_of_freedom)
    f_p_value = f_test.p_values

    df_model = model.num_predictors
    df_residual = model.degrees_of_freedom

    generate_linear_regression_report(
        coefficients,
        std_errors,
        t_values,
        t_test_p_values,
        residual_summary,
        r_squared,
        adjusted_r_squared,
        residual_standard_error,
        f_statistic,
        f_p_value,
        df_residual,
        df_model,
        lr_model=model
    )

