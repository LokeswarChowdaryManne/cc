import numpy as np
import matplotlib.pyplot as plt

close_prices=[29,20,25,29,31,33,34,27,26,30,
            29,28,28,26,27,26,30,28,26,30,
              31,30,37,30,33,31,27,33,37,29,
              28,30,29,34,30,20,17,23,24,34,
              36,35,33,29,25,27,30,29,28,32]

def mean(series):
    return sum(series) / len(series)

def acf(series, lag):
    n = len(series)
    x_bar = mean(series)
    num = 0
    denom = 0
    for t in range(lag, n):
        num += (series[t] - x_bar) * (series[t - lag] - x_bar)
    for t in range(n):
        denom += (series[t] - x_bar) ** 2
    return num / denom

def acf_all(series, max_lag):
    return [acf(series, k) for k in range(max_lag+1)]

def calculate_pacf(series, max_lag):
    rho = acf_all(series, max_lag)   # autocorrelations
    pacf_vals = [1.0]  # PACF(0) = 1

    for k in range(1, max_lag+1):
        # build Toeplitz matrix of autocorrelations
        P_k = np.array([[rho[abs(i-j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k+1])
        phi_k = np.linalg.solve(P_k, rho_k)  # solve P_k * phi_k = rho_k
        pacf_vals.append(phi_k[-1])  # last coefficient = PACF(k)

    return pacf_vals

print("\nACF (first 10 lags):")
print(acf_all(close_prices, 10))

print("\nPACF (first 10 lags):")
print(calculate_pacf(close_prices, 10))

conf=1.96/np.sqrt(len(close_prices))

def plot_acf(series, max_lag):
    acf_vals = acf_all(series, max_lag)
    lags = list(range(max_lag+1))
    plt.stem(lags, acf_vals)
    plt.xlabel("Lag")
    plt.axhline(y=conf, color='r', linestyle='--')
    plt.axhline(y=-conf, color='r', linestyle='--')
    plt.ylabel("ACF")
    plt.title("Autocorrelation Function")
    plt.show()

def plot_pacf(series, max_lag):
    pacf_vals = calculate_pacf(series, max_lag)
    lags = list(range(max_lag+1))
    plt.stem(lags, pacf_vals)
    plt.xlabel("Lag")
    plt.axhline(y=conf, color='r', linestyle='--')
    plt.axhline(y=-conf, color='r', linestyle='--')
    plt.ylabel("PACF")
    plt.title("Partial Autocorrelation Function")
    plt.show()

plot_acf(close_prices, 25)
plot_pacf(close_prices, 20)

print("PACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'PACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

conf_level = 1.96 / np.sqrt(len(close_prices)) # Assuming conf_level is defined

pacf_vals = calculate_pacf(close_prices, 20) # Assuming calculate_pacf is defined and max_lag is 20

for lag, val in enumerate(pacf_vals):
    if lag == 0:
        significant = "N/A"
        decision = "PACF(0) = 1 (by definition)"
    else:
        # Null hypothesis: PACF(lag) = 0
        # Reject H0 if |PACF| > 2/root(n)
        if abs(val) > conf_level:
            significant = "Yes"
            decision = "Reject H0: Significant partial autocorrelation"
        else:
            significant = "No"
            decision = "Fail to reject H0: Not significant"

    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")