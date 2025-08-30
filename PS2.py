#Newton Interpolation and lagrange Interpolation
# Analytical function
def analytical_e(T):
    return 0.02424 * (T / 303.16)**1.27591

# Newton's Divided Difference Interpolation
def divided_diff(X, Y):
    n = len(Y)
    coef = np.zeros([n, n])
    coef[:, 0] = Y

    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (X[i+j] - X[i])

    return coef[0, :]

def newton_interp(x, X, coef):
    n = len(X)
    result = coef[0]

    for i in range(1, n):
        term = coef[i]

        for j in range(i):
            term *= (x - X[j])

        result += term

    return result

# Lagrange Interpolation
def lagrange_interp(x, X, Y):
    total = 0
    n = len(X)

    for i in range(n):
        term = Y[i]

        for j in range(n):
            if i != j:
                term *= (x - X[j]) / (X[i] - X[j])

        total += term

    return total

coefficients = divided_diff(T, e)

# Values to interpolate
x_vals = np.linspace(300, 2000, 500)

newton_values = [newton_interp(x, T, coefficients) for x in x_vals]
lagrange_values = [lagrange_interp(x, T, e) for x in x_vals]
analytical_values = [analytical_e(x) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(T, e, 'o', label='Given Data')
plt.plot(x_vals, newton_values, '--', label="Newton's Interpolation")
plt.plot(x_vals, lagrange_values, ':', label="Lagrange Interpolation")
plt.plot(x_vals, analytical_values, '-', label='Analytical Expression')
plt.xlabel('Temperature (T)')
plt.ylabel('Emittance e(T)')
plt.legend()
plt.title('Emittance vs Temperature')
plt.grid(True)
plt.show()

x_test_vals = [0.5, 3]
print("f(x) at x = 0.5 using Newton:", newton_interp(0.5, T, coefficients))
print("f(x) at x = 0.5 using Lagrange:", lagrange_interp(0.5, T, e))
print("f(x) at x = 3 using Newton:", newton_interp(3, T, coefficients))
print("f(x) at x = 3 using Lagrange:", lagrange_interp(3, T, e))

#--------------------------------------------------------------------------------------------------------

#Newton Forward interpolation
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

force = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
stretch = np.array([19, 57, 94, 134, 173, 216, 256, 297, 343])

h = force[1] - force[0]

# Newton forward difference table
def forward_difference_table(y):
    n = len(y)
    diff_table = np.zeros((n, n))
    diff_table[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            diff_table[i][j] = diff_table[i+1][j-1] - diff_table[i][j-1]
    return diff_table

# Newton forward interpolation
def newton_forward(x, x0, h, diff_table, n_terms=4):
    u = (x - x0) / h
    result = diff_table[0][0]
    u_term = 1
    fact = 1
    for i in range(1, n_terms):
        u_term *= (u - i + 1)
        fact *= i
        result += (u_term * diff_table[0][i]) / fact
    return result

diff_table = forward_difference_table(stretch)

# Interpolate for values: 15, 17, 85
query_points = [15, 17, 85]
interpolated_values = [newton_forward(x, force[0], h, diff_table, len(force)) for x in query_points]

for i, x in enumerate(query_points):
    print(f"Interpolated stretch at force {x} = {interpolated_values[i]:.2f}")

# (a) Plot original and interpolated points
x_dense = np.linspace(10, 90, 500)
y_interp = [newton_forward(x, force[0], h, diff_table, len(force)) for x in x_dense]

plt.figure(figsize=(10, 6))
plt.plot(force, stretch, 'o', label='Original Data')
plt.plot(x_dense, y_interp, '--', label='Newton Forward Interpolation')
plt.scatter(query_points, interpolated_values, c='red', label='Interpolated Points')
plt.xlabel('Force')
plt.ylabel('Stretch')
plt.title('Stretch vs Force using Newton Forward Interpolation')
plt.legend()
plt.grid(True)
plt.show()

# (b) Calculate and plot errors (on original points)

stretch_pred = [newton_forward(x, force[0], h, diff_table, len(force)) for x in force]
errors = stretch_pred - stretch

plt.figure(figsize=(10, 4))
plt.stem(force, errors)
plt.title("Interpolation Error at Given Data Points")
plt.xlabel("Force")
plt.ylabel("Error (Predicted - Actual)")
plt.grid(True)
plt.show()

x_dense = np.linspace(10, 90, 500)
stretch_pred_dense = [newton_forward(x, force[0], h, diff_table, len(force)) for x in x_dense]
stretch_true_dense = np.interp(x_dense, force, stretch)
errors_dense = np.array(stretch_pred_dense) - stretch_true_dense

plt.figure(figsize=(10, 4))
plt.plot(x_dense, errors_dense, label="Interpolation Error", color='red')
plt.title("Interpolation Error at Interior Points [10, 90]")

plt.xlabel("Force")
plt.ylabel("Error (Predicted - Actual)")

plt.grid(True)
plt.legend()
plt.show()


# (c) T-test to check model fitting (paired t-test)

t_stat, p_value = ttest_rel(stretch, stretch_pred)
print(f"T-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

if p_value > 0.05:
    print("Interpolating polynomial reasonably fits the data (fail to reject H0)")
else:
    print("Interpolating polynomial may not fit well (reject H0)")


#----------------------------------------------------------------------------------------------------

#Cubic Spline

import random
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

"""
n = 20
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
f = [3, 4, 7, 1, 5, 9, 2, 6, 3, 8,4, 7, 0, 5, 6, 2, 4, 1, 9, 3]
"""

n = 4
x = [3,4.5,7,9]
f = [2.5,1,2.5,0.5]

def spline_equation_row(xi_minus1, xi, xi_plus1, f_xi_minus1, f_xi, f_xi_plus1):
    a = xi - xi_minus1
    b = 2 * (xi_plus1 - xi_minus1)
    c = xi_plus1 - xi
    rhs = (6 / (xi_plus1 - xi)) * (f_xi_plus1 - f_xi) - (6 / (xi - xi_minus1)) * (f_xi - f_xi_minus1)
    return a, b, c, rhs

variables = sp.symbols(f'f\'\'1:{n-1}')
equations = []

for i in range(1, n - 1):
    a, b, c, rhs = spline_equation_row(x[i - 1], x[i], x[i + 1], f[i - 1], f[i], f[i + 1])
    row = 0
    if i > 1:
        row += a * variables[i - 2]
    row += b * variables[i - 1]
    if i < n - 2:
        row += c * variables[i]
    equations.append(sp.Eq(row, rhs))

sol = sp.solve(equations, variables)

fpp = [0] + [sol[v] for v in variables] + [0]

def cubic_spline_interpolation(x_val, xi, xi1, fxi, fxi1, fppi, fppi1):
    h = xi1 - xi
    term1 = fppi * ((xi1 - x_val) ** 3) / (6 * h)
    term2 = fppi1 * ((x_val - xi) ** 3) / (6 * h)
    term3 = (fxi / h - fppi * h / 6) * (xi1 - x_val)
    term4 = (fxi1 / h - fppi1 * h / 6) * (x_val - xi)
    #print(f"{term1}+{term2}+{term3}+{term4}")
    return term1 + term2 + term3 + term4

X_vals = []
Y_vals = []

for i in range(n - 1):
    xs = np.linspace(x[i],x[i+1],100)
    ys = [cubic_spline_interpolation(xv, x[i], x[i + 1], f[i], f[i + 1], fpp[i], fpp[i + 1]) for xv in xs]
    X_vals.extend(xs)
    Y_vals.extend(ys)


plt.figure(figsize=(12, 6))
plt.plot(X_vals, Y_vals, label="Cubic Spline", color="blue")
plt.plot(x, f, 'ro', label="Data Points")
plt.title("Cubic Spline Interpolation")
plt.legend()
plt.grid(True)

plt.show()

