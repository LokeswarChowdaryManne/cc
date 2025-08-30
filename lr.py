#q1 is intnerpol

#fw
#QN 2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

force = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
stretch = np.array([19, 57, 94, 134, 173, 216, 256, 297, 343])

def newton_forward_interpolation(x, y, x_eval):
    n = len(x)
    diff_table = np.zeros((n, n))
    diff_table[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i, j] = diff_table[i + 1, j - 1] - diff_table[i, j - 1]

    h = x[1] - x[0]
    result = []
    for x0 in x_eval:
        u = (x0 - x[0]) / h
        p = diff_table[0, 0]
        u_term = 1
        for k in range(1, n):
            u_term *= (u - (k - 1)) / k
            p += diff_table[0, k] * u_term
        result.append(p)
    return np.array(result)

x_eval = np.array([15, 17, 85])
interpolated = newton_forward_interpolation(force, stretch, force)
x_fine = np.linspace(10, 90, 100)
y_fine = newton_forward_interpolation(force, stretch, x_fine)

plt.figure(figsize=(10,5))
plt.plot(force, stretch, 'o', label='Data Points')
plt.plot(x_fine, y_fine, '-', label='Newton Interpolation')
plt.plot(x_ eval, newton_forward_interpolation(force, stretch, x_eval), 's', label='Interpolated Points')
plt.xlabel('Force')
plt.ylabel('Stretch')
plt.title('Newton Forward Interpolation')
plt.grid(True)
plt.legend()
plt.show()

print("Interpolated stretch values:")
for f, s in zip(x_eval, newton_forward_interpolation(force, stretch, x_eval)):
    print(f"Force {f}: Stretch {s:.2f}")

interpolated_stretch = newton_forward_interpolation(force, stretch, force)
errors = stretch - interpolated_stretch

plt.figure(figsize=(10, 5))
plt.plot(force, errors, 'o-', label='Interpolation Errors')
plt.xlabel('Force')
plt.ylabel('Error')
plt.title('Interpolation Errors vs Force')
plt.grid(True)
plt.legend()
plt.show()

# Part (c): T-test
t_stat, p_value = ttest_rel(stretch, interpolated_stretch)
print("T-test results:")
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
print("Fit analysis: P-value > 0.05 suggests the polynomial fits the data well (no significant difference).")



#dd and lagrange
#QN 3
import numpy as np
import matplotlib.pyplot as plt

T = np.array([300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
e = np.array([0.024, 0.035, 0.046, 0.058, 0.067, 0.083, 0.097, 0.111, 0.125, 0.140, 0.155, 0.170, 0.186, 0.202, 0.219, 0.235, 0.252, 0.269])

def newton_divided_diff(x, y, x_eval):
    n = len(x)
    coef = np.zeros(n)
    coef[0] = y[0]
    F = np.zeros((n, n))
    F[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])
        coef[j] = F[0, j]

    result = []
    for x0 in x_eval:
        p = coef[0]
        prod = 1
        for j in range(1, n):
            prod *= (x0 - x[j - 1])
            p += coef[j] * prod
        result.append(p)
    return np.array(result)

def lagrange_interpolation(x, y, x_eval):
    n = len(x)
    result = []
    for x0 in x_eval:
        p = 0
        for i in range(n):
            term = y[i]
            for j in range(n):
                if j != i:
                    term *= (x0 - x[j]) / (x[i] - x[j])
            p += term
        result.append(p)
    return np.array(result)

x_eval = np.array([500, 300])

newton_vals = newton_divided_diff(T, e, x_eval)
print("Newton's Divided Difference Interpolation:")
for t, val in zip(x_eval, newton_vals):
    print(f"Temperature {t} K: Emittance {val:.4f}")

lagrange_vals = lagrange_interpolation(T, e, x_eval)
print("\nLagrange Interpolation:")
for t, val in zip(x_eval, lagrange_vals):
    print(f"Temperature {t} K: Emittance {val:.4f}")

T_fine = np.linspace(300, 2000, 200)
newton_fine = newton_divided_diff(T, e, T_fine)
lagrange_fine = lagrange_interpolation(T, e, T_fine)

plt.figure(figsize=(10, 6))
plt.plot(T, e, 'o', label='Original Data', markersize=8)
plt.plot(T_fine, newton_fine, '-', label='Newton Interpolation', linewidth=2)
plt.plot(T_fine, lagrange_fine, '--', label='Lagrange Interpolation', linewidth=2)
plt.xlabel('Temperature (K)')
plt.ylabel('Emittance')
plt.title('Emittance vs. Temperature: Newton and Lagrange Interpolation')
plt.grid(True)
plt.legend()
plt.show()

print("\nComparison with Original Data:")
print("The interpolated polynomials pass through all original data points, as expected for interpolation.")
print("Difference between Newton and Lagrange at evaluated points:", np.abs(newton_vals - lagrange_vals))
print("The small differences (if any) are due to numerical precision. Both polynomials are theoretically identical.")

