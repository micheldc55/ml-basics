"""Polynomial regression is just a linear regression model with polynomial features.

Given X = [[1, 2], [2, 3], [3, 4]] and y = [2, 3, 4], we can transform X to include polynomial features
of degree 2 as follows:

X = [[1, 2, 1, 4, 2, 4], [2, 3, 4, 9, 6, 12], [3, 4, 9, 16, 12, 16]]

For row i in X, the polynomial features are [1, x_i1, x_i1^2, x_i2, x_i1*x_i2, x_i2^2].
"""