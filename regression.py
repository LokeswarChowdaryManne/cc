# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 15:04:41 2025

@author: shwet
"""
#thiru code


# linear regression
#5.------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('500040.csv') # Aditya_birla Real Estate Industries

dates = np.random.randint(1, len(df), 150)

selected_df = df.iloc[dates].copy()

selected_df.loc[:, 'Rate of Return'] = ((selected_df['Close Price'] - selected_df['Open Price']) / selected_df['Open Price']) * 100

f_df = selected_df[['Open Price', 'Close Price', 'Rate of Return']].copy()
print(f_df.head())

y = f_df['Rate of Return'].values

x = np.arange(len(f_df))

mean_x = np.mean(x)
mean_y = np.mean(y)

numerator_linear = np.sum((x - mean_x) * (y - mean_y))
denominator_linear = np.sum((x - mean_x)**2)
slope_linear = numerator_linear / denominator_linear
intercept_linear = mean_y - slope_linear * mean_x

print(f"Linear Regression Equation: y = {slope_linear:.4f}x + {intercept_linear:.4f}")

y_pred_linear = slope_linear * x + intercept_linear

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(x, y_pred_linear, color='red', label='Linear Regression Fit')
plt.xlabel("Data Point Index")
plt.ylabel("Rate of Return")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# Quadractic Regression

x_quadratic = np.column_stack((np.ones(len(x)), x, x**2))

# (X^T * X)^-1 * X^T * y
X_transpose_X = np.dot(x_quadratic.T, x_quadratic)
X_transpose_y = np.dot(x_quadratic.T, y)

X_transpose_X_inv = np.linalg.inv(X_transpose_X)

coefficients_quadratic = np.dot(X_transpose_X_inv, X_transpose_y)

intercept_quadratic = coefficients_quadratic[0]
slope_quadratic_x = coefficients_quadratic[1]
slope_quadratic_x2 = coefficients_quadratic[2]

print(f"Quadratic Regression Equation: y = {slope_quadratic_x2:.4f}x^2 + {slope_quadratic_x:.4f}x + {intercept_quadratic:.4f}")

y_pred_quadratic = intercept_quadratic + slope_quadratic_x * x + slope_quadratic_x2 * (x**2)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(x, y_pred_quadratic, color='red', label='Quadratic Regression Fit')
plt.xlabel("Data Point Index")
plt.ylabel("Rate of Return")
plt.title("Quadratic Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# cubic regression

x_cubic = np.column_stack((np.ones(len(x)), x, x**2, x**3))

# (X^T * X)^-1 * X^T * y
X_transpose_X = np.dot(x_cubic.T, x_cubic)
X_transpose_y = np.dot(x_cubic.T, y)

X_transpose_X_inv = np.linalg.inv(X_transpose_X)

coefficients_cubic = np.dot(X_transpose_X_inv, X_transpose_y)

intercept_cubic = coefficients_cubic[0]
slope_cubic_x = coefficients_cubic[1]
slope_cubic_x2 = coefficients_cubic[2]
slope_cubic_x3 = coefficients_cubic[3]

print(f"Cubic Regression Equation: y = {slope_cubic_x3:.4f}x^3 + {slope_cubic_x2:.4f}x^2 + {slope_cubic_x:.4f}x + {intercept_cubic:.4f}")

y_pred_cubic = intercept_cubic + slope_cubic_x * x + slope_cubic_x2 * (x**2) + slope_cubic_x3 * (x**3)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Original Data')
plt.plot(x, y_pred_cubic, color='red', label='Cubic Regression Fit')
plt.xlabel("Data Point Index")
plt.ylabel("Rate of Return")
plt.title("Cubic Regression Fit")
plt.legend()
plt.grid(True)
plt.show()

# Validation n Coefficient of Determination

mse_linear = np.mean((y - y_pred_linear)**2)
# R-squared
ssr_linear = np.sum((y_pred_linear - mean_y)**2)
sst_linear = np.sum((y - mean_y)**2)
r2_linear = ssr_linear / sst_linear

print(f"--- Linear Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_linear:.4f}")
print(f"R-squared (R2): {r2_linear:.4f}")
print("-" * 30)

# evaluation metrics for Quadratic Regression
mse_quadratic = np.mean((y - y_pred_quadratic)**2)
# R-squared
ssr_quadratic = np.sum((y_pred_quadratic - mean_y)**2)
sst_quadratic = np.sum((y - mean_y)**2)
r2_quadratic = ssr_quadratic / sst_quadratic

print(f"--- Quadratic Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_quadratic:.4f}")
print(f"R-squared (R2): {r2_quadratic:.4f}")
print("-" * 30)


# evaluation metrics for Cubic Regression
mse_cubic = np.mean((y - y_pred_cubic)**2)
# R-squared
ssr_cubic = np.sum((y_pred_cubic - mean_y)**2)
sst_cubic = np.sum((y - mean_y)**2)
r2_cubic = ssr_cubic / sst_cubic

print(f"--- Cubic Regression Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_cubic:.4f}")
print(f"R-squared (R2): {r2_cubic:.4f}")
print("-" * 30)

#lower MSE[0,inf) and higher R-squared[0,1] generally indicates a better fit.


from scipy import stats

residuals_linear = y - y_pred_linear
print("Linear Regression Residuals (first 10):")
print(residuals_linear[:10])

ttest_linear = stats.ttest_1samp(residuals_linear, 0)
print(f"\nLinear Regression Residuals T-test: statistic={ttest_linear.statistic}, pvalue={ttest_linear.pvalue}")

residuals_quadratic = y - y_pred_quadratic
print("\nQuadratic Regression Residuals (first 10):")
print(residuals_quadratic[:10])

ttest_quadratic = stats.ttest_1samp(residuals_quadratic, 0)
print(f"\nQuadratic Regression Residuals T-test: statistic={ttest_quadratic.statistic}, pvalue={ttest_quadratic.pvalue}")

residuals_cubic = y - y_pred_cubic
print("\nCubic Regression Residuals (first 10):")
print(residuals_cubic[:10])

ttest_cubic = stats.ttest_1samp(residuals_cubic, 0)
print(f"\nCubic Regression Residuals T-test: statistic={ttest_cubic.statistic}, pvalue={ttest_cubic.pvalue}")
