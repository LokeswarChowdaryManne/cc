#PS4
#Euler Method

import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.stats import ttest_ind


def dy_dx(x, y):
    return x + y


def euler_method(x0, y0, h, n_steps):
    x_values = [x0]
    y_values = [y0]

    for _ in range(n_steps):
        x_new = x_values[-1] + h
        y_new = y_values[-1] + h * dy_dx(x_values[-1], y_values[-1])
        x_values.append(x_new)
        y_values.append(y_new)

    return x_values, y_values


x0, xn = 0, 1
y0 = 1
h = 1/14
n_steps = int((xn - x0) / h)


x_euler, y_euler = euler_method(x0, y0, h, n_steps)


x_exact = np.linspace(x0, xn, 100)
y_exact = 2 * np.exp(x_exact) - x_exact - 1


y_exact_at_euler = 2 * np.exp(x_euler) - x_euler - 1


t_stat, p_value = ttest_ind(y_euler, y_exact_at_euler)


plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'b-', label='Exact Solution', linewidth=2)
plt.plot(x_euler, y_euler, 'ro--', label='Euler Method', markersize=6)
plt.title(f"Solution of $\\frac{{dy}}{{dx}} = x + y$\n(T-test p-value: {p_value:.3e})", fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()

table = PrettyTable()
table.field_names = ["Step", "x", "Euler (y)", "Exact (y)", "Error"]
for i, (x, y_num) in enumerate(zip(x_euler, y_euler)):
    y_true = 2 * np.exp(x) - x - 1
    error = abs(y_num - y_true)
    table.add_row([i, round(x, 4), round(y_num, 6), round(y_true, 6), round(error, 6)])
print(table)

print("\nT-test Results:")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

if p_value < 0.05:
    print("Conclusion: Euler's method results are significantly different from the exact solution (p < 0.05)")
else:
    print("Conclusion: No significant difference between Euler's method and exact solution (p ≥ 0.05)")

#----------------------------------------------------------------------------------------------------------------------------------------------------

#Runge Kutta method
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from scipy.stats import ttest_rel


def dy_dx(x, y):
    return y - x

def rk4_method(x0, y0, h, n_steps):
    x_values = [x0]
    y_values = [y0]

    for _ in range(n_steps):
        x = x_values[-1]
        y = y_values[-1]

        k1 = h * dy_dx(x, y)
        k2 = h * dy_dx(x + h/2, y + k1/2)
        k3 = h * dy_dx(x + h/2, y + k2/2)
        k4 = h * dy_dx(x + h, y + k3)

        y_new = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_new = x + h

        x_values.append(x_new)
        y_values.append(y_new)

    return x_values, y_values

x0, xn = 0, 1
y0 = 2
h = 1/14
n_steps = int((xn - x0) / h)


x_rk4, y_rk4 = rk4_method(x0, y0, h, n_steps)


x_exact = np.linspace(x0, xn, 100)
y_exact = np.exp(x_exact) + x_exact + 1

y_exact_at_rk4 = np.exp(x_rk4) + x_rk4 + 1

t_stat, p_value = ttest_rel(y_rk4, y_exact_at_rk4)

plt.figure(figsize=(10, 6))
plt.plot(x_exact, y_exact, 'b-', label='Exact Solution', linewidth=2)
plt.plot(x_rk4, y_rk4, 'ro--', label=f'RK4 Method (h={h:.3f})', markersize=6)
plt.title(f"Solution of $\\frac{{dy}}{{dx}} = y - x$\n(Paired t-test p-value: {p_value:.3e})", fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.show()

table = PrettyTable()
table.field_names = ["Step", "x", "RK4 (y)", "Exact (y)", "Error", "Relative Error (%)"]
for i, (x, y_num, y_true) in enumerate(zip(x_rk4, y_rk4, y_exact_at_rk4)):
    error = abs(y_num - y_true)
    rel_error = 100 * error / y_true if y_true != 0 else float('nan')
    table.add_row([
        i,
        round(x, 4),
        round(y_num, 6),
        round(y_true, 6),
        round(error, 6),
        round(rel_error, 4)
    ])
print(table)

print("\nStatistical Comparison:")
print(f"t-statistic = {t_stat:.4f}, p-value = {p_value:.4e}")

if p_value < 0.05:
    print("Significant difference between RK4 and exact solution (p < 0.05)")
else:
    print("No significant difference (p ≥ 0.05)")

mean_abs_error = np.mean(np.abs(y_rk4 - y_exact_at_rk4))
max_error = np.max(np.abs(y_rk4 - y_exact_at_rk4))

print(f"\nError Analysis:\nMean Absolute Error = {mean_abs_error:.6f}\nMaximum Error = {max_error:.6f}")
