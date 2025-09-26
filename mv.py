import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

year   = [1991, 1992, 1993, 1994, 1995]
spring = [102, 110, 111, 115, 122]
summer = [120, 126, 128, 135, 144]
fall   = [90, 95, 97, 103, 110]
winter = [78, 83, 86, 91, 98]

seasonal_df = pd.DataFrame({
    'year': year,
    'spring': spring,
    'summer': summer,
    'fall': fall,
    'winter': winter
})

seasonal_ts = seasonal_df.melt(id_vars='year', var_name='season', value_name='value')

seasonal_order = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
seasonal_ts['quarter'] = seasonal_ts['season'].map(seasonal_order)
seasonal_ts = seasonal_ts.sort_values(by=['year', 'quarter']).reset_index(drop=True)
print("\nData:")
print(seasonal_ts)

seasonal_ts['4_point_moving_avg'] = (
    seasonal_ts['value']
    .rolling(window=4, center=True).mean()
    .rolling(window=2, center=True).mean()
)

seasonal_ts['percent_of_moving_avg'] = (seasonal_ts['value'] / seasonal_ts['4_point_moving_avg']) * 100
print("\n", seasonal_ts[['year', 'season', 'value', '4_point_moving_avg', 'percent_of_moving_avg']])

modified_seasonal_indices = seasonal_ts.groupby('season')['percent_of_moving_avg'].median()
normalization_factor = 400 / modified_seasonal_indices.sum()
seasonal_indices = modified_seasonal_indices * normalization_factor
print("\nSeasonal Indices:")
print(seasonal_indices)

seasonal_ts['seasonal_index'] = seasonal_ts['season'].map(seasonal_indices)

seasonal_ts['deseasonalized_value'] = (seasonal_ts['value'] / seasonal_ts['seasonal_index']) * 100
print("\n", seasonal_ts[['year', 'season', 'value', 'seasonal_index', 'deseasonalized_value']])


deseasonalized_ts_clean = seasonal_ts.dropna(subset=['deseasonalized_value']).copy()
t = np.arange(len(deseasonalized_ts_clean))
y = deseasonalized_ts_clean['deseasonalized_value'].values
n = len(t)

t_mean, y_mean = np.mean(t), np.mean(y)
b_lin = np.sum((t - t_mean) * (y - y_mean)) / np.sum((t - t_mean)**2)
a_lin = y_mean - b_lin*t_mean
y_pred_lin = a_lin + b_lin*t

Sx = np.sum(t)
Sx2 = np.sum(t**2)
Sx3 = np.sum(t**3)
Sx4 = np.sum(t**4)
Sy = np.sum(y)
Sxy = np.sum(t*y)
Sx2y = np.sum((t**2)*y)

A = np.array([[n, Sx, Sx2],
              [Sx, Sx2, Sx3],
              [Sx2, Sx3, Sx4]])
B = np.array([Sy, Sxy, Sx2y])
a, b, c = np.linalg.solve(A, B)
y_pred_quad = a + b*t + c*(t**2)

logy = np.log(y)
logy_mean = np.mean(logy)
b_exp = np.sum((t - t_mean) * (logy - logy_mean)) / np.sum((t - t_mean)**2)
a_exp = logy_mean - b_exp*t_mean
A_exp = np.exp(a_exp)
y_pred_exp = A_exp * np.exp(b_exp*t)

print("\nTrend Equations:")
print(f"Linear: y = {a_lin:.4f} + {b_lin:.4f} * t")
print(f"Quadratic: y = {a:.4f} + {b:.4f} * t + {c:.4f} * t^2")
print(f"Exponential: y = {A_exp:.4f} * e^({b_exp:.4f} * t)")

def paired_ttest(y_true, y_pred):
    d = y_true - y_pred
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)
    n = len(d)
    t_stat = mean_d / (std_d / np.sqrt(n))
    #p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    return t_stat

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res/ss_tot

models = {
    "Linear": y_pred_lin,
    "Quadratic": y_pred_quad,
    "Exponential": y_pred_exp
}

results = {}
for model, y_pred in models.items():
    t_stat= paired_ttest(y, y_pred)
    r2 = r_squared(y, y_pred)
    results[model] = {"t": t_stat, "R2": r2}

print("Trend Model Results:")
for model, vals in results.items():
    print(f"{model}: t={vals['t']:.4f}, R2={vals['R2']:.4f}")

best_model = max(results, key=lambda m: results[m]["R2"])
print("\nBest model:", best_model)
best_pred = models[best_model]

deseasonalized_ts_clean['cyclical_variation'] = deseasonalized_ts_clean['deseasonalized_value'] - best_pred
deseasonalized_ts_clean['relative_cyclical_residual'] = (
    deseasonalized_ts_clean['cyclical_variation'] / best_pred
) * 100

print("\nCyclical Variation:")
print(deseasonalized_ts_clean[['year', 'season', 'cyclical_variation']])

print("\nRelative Cyclical Residual (%):")
print(deseasonalized_ts_clean[['year', 'season', 'relative_cyclical_residual']])

plt.figure(figsize=(12,6))
plt.plot(deseasonalized_ts_clean.index, deseasonalized_ts_clean['deseasonalized_value'], 'o-', label='Deseasonalized Data', color='green')
plt.plot(deseasonalized_ts_clean.index, y_pred_lin, '--', label='Linear Trend', color='blue')
plt.plot(deseasonalized_ts_clean.index, y_pred_quad, '--.', label='Quadratic Trend', color='black')
plt.plot(deseasonalized_ts_clean.index, y_pred_exp, '--', label='Exponential Trend', color='red')
plt.plot(deseasonalized_ts_clean.index, best_pred, '-', label='Best Fit', linewidth=2, color='purple')
plt.title(f"Original & Deseasonalized Data with Trend Fits")
plt.xlabel("Time Index")
plt.ylabel("Deseasonalized Value")
plt.legend()
plt.grid(True)
plt.show()
