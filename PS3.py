import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def f(x):
  return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

def trapezoidal_rule(func, a, b, n_points):
  n_intervals = n_points - 1
  x = np.linspace(a, b, n_points)
  y = func(x)
  h = (b - a) / n_intervals
  integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
  return integral

def simpsons_rule(func, a, b, n_points):
  n_intervals = n_points - 1
  if n_intervals % 2 != 0:
    raise ValueError("Number of intervals must be even")

  x = np.linspace(a, b, n_points)
  y = func(x)
  h = (b - a) / n_intervals
  integral = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
  return integral

true_value = 0.977249868051821
a = -10.0
b = 2.0

print("Standard Normal Distribution Integration")

n1 = 2001
print(f"Using {n1} points")

trap_result_1 = trapezoidal_rule(f, a, b, n1)
trap_error_1 = abs(trap_result_1 - true_value)
print(f"Trapezoidal Rule Result: {trap_result_1:.15f}")
print(f"Absolute Error:          {trap_error_1:.15e}")

simp_result_1 = simpsons_rule(f, a, b, n1)
simp_error_1 = abs(simp_result_1 - true_value)
print(f"Simpson's Rule Result:   {simp_result_1:.15f}")
print(f"    Absolute Error:          {simp_error_1:.15e}")

n2 = 4001
print(f"Using {n2} points")
trap_result_2 = trapezoidal_rule(f, a, b, n2)
trap_error_2 = abs(trap_result_2 - true_value)
print(f"Trapezoidal Rule Result: {trap_result_2}")
print(f"Absolute Error: {trap_error_2}")

simp_result_2 = simpsons_rule(f, a, b, n2)
simp_error_2 = abs(simp_result_2 - true_value)
print(f"Simpson's Rule Result:   {simp_result_2:.15f}")
print(f"Absolute Error:          {simp_error_2:.15e}")

print("Convergence Analysis")
trap_error_ratio = trap_error_1 / trap_error_2
simp_error_ratio = simp_error_1 / simp_error_2
