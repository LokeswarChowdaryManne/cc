#bio
def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    n, m = len(seq1), len(seq2)

    #Create score matrix
    score = [[0]*(m+1) for _ in range(n+1)]

    #Initialize first row & column
    for i in range(1, n+1):
        score[i][0] = i * gap
    for j in range(1, m+1):
        score[0][j] = j * gap

    # Fill score matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            diag = score[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            up   = score[i-1][j] + gap
            left = score[i][j-1] + gap
            score[i][j] = max(diag, up, left)

    # Print score matrix
    print("\nSCORE MATRIX:")
    print("     ", "   ".join(["-"] + list(seq2)))
    for i in range(n + 1):
        row = "-" if i == 0 else seq1[i - 1]
        vals = "   ".join(f"{x:>3}" for x in score[i])
        print(f"{row}  {vals}")

    # Traceback for alignment
    align1, align2, i, j = "", "", n, m
    while i > 0 and j > 0:
        current = score[i][j]
        diag, up, left = score[i-1][j-1], score[i-1][j], score[i][j-1]
        if current == diag + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1, align2, i, j = seq1[i-1] + align1, seq2[j-1] + align2, i-1, j-1
        elif current == up + gap:
            align1, align2, i = seq1[i-1] + align1, "-" + align2, i-1
        else:
            align1, align2, j = "-" + align1, seq2[j-1] + align2, j-1

    while i > 0:
        align1, align2, i = seq1[i-1] + align1, "-" + align2, i-1
    while j > 0:
        align1, align2, j = "-" + align1, seq2[j-1] + align2, j-1

    # Print result with visual alignment
    print("\nALIGNMENT RESULT:")
    connector = ""
    for a, b in zip(align1, align2):
        if a == b:
            connector += "|"
        elif a == "-" or b == "-":
            connector += " "
        else:
            connector += ":"

    print(align1)
    print(connector)
    print(align2)
    print("Final Score:", score[n][m])

    return align1, align2, score[n][m]


# Example
needleman_wunsch("TGGTG", "ATCGT")

#code1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ LOAD CSV DATA
# ------------------------------------------------------------
companies = ["500040", "531158", "500530 (1)"]
N_sim = 5000

open_prices = pd.DataFrame()
close_prices = pd.DataFrame()

for c in companies:
    df = pd.read_csv(f"{c}.csv", parse_dates=["Month"])
    df.set_index("Month", inplace=True)
    open_prices[c] = df["Open Price"]
    close_prices[c] = df["Close Price"]

# ------------------------------------------------------------
# 2️⃣ CALCULATE MONTHLY RETURNS (k1, k2, k3, k4)
# ------------------------------------------------------------
returns = (close_prices - open_prices) / open_prices
returns = returns.dropna()
returns.columns = ["k1", "k2", "k3"]

# ------------------------------------------------------------
# 3️⃣ MEAN (μ) AND STANDARD DEVIATION (σ)
# ------------------------------------------------------------
mu = returns.mean().values.reshape(-1, 1)
cov_matrix = returns.cov().values
sigma = np.sqrt(np.diag(cov_matrix))

mu1, mu2, mu3 = mu.flatten()
std1, std2, std3 = sigma

print("\n" + "="*75)
print("📈  ASSET-WISE STATISTICS")
print("="*75)
print(f"{'Asset':<15}{'Mean (μ)':>15}{'Std Dev (σ)':>20}")
print("-"*75)
for i, c in enumerate(companies):
    print(f"{c:<15}{mu[i,0]:>15.6f}{sigma[i]:>20.6f}")
print("="*75)

print("\nCovariance Matrix (Σ):")
print("-"*75)
print(pd.DataFrame(np.round(cov_matrix, 6), index=companies, columns=companies))
print("\nCorrelation Matrix (ρ):")
print("-"*75)
print(pd.DataFrame(np.round(np.corrcoef(returns.T), 4), index=companies, columns=companies))
print("="*75)

# ------------------------------------------------------------
# 4️⃣ RANDOM WEIGHTS (SUM TO 1)
# ------------------------------------------------------------
def random_weights(n_assets, n_portfolios):
    w = np.random.rand(n_portfolios, n_assets)
    w = w / w.sum(axis=1)[:, None]
    return w

weights = random_weights(len(companies), N_sim)

# ------------------------------------------------------------
# 5️⃣ PORTFOLIO MEAN (μv) AND VARIANCE (σv²)
# ------------------------------------------------------------
mu_v = weights @ mu
var_v = np.array([w @ cov_matrix @ w.T for w in weights])
std_v = np.sqrt(var_v)

# ------------------------------------------------------------
# 6️⃣ CLOSED-FORM MINIMUM VARIANCE PORTFOLIO (S₀)
# ------------------------------------------------------------
ones = np.ones((len(companies), 1))
inv_cov = np.linalg.inv(cov_matrix)
w_minvar = (inv_cov @ ones) / (ones.T @ inv_cov @ ones)
w_minvar = w_minvar.flatten()

mu_min = float(w_minvar @ mu)
var_min = float(w_minvar @ cov_matrix @ w_minvar)
std_min = np.sqrt(var_min)

print("\n\n" + "="*75)
print("🎯  CLOSED-FORM MINIMUM VARIANCE PORTFOLIO  (S₀)")
print("="*75)
print(f"{'Asset':<15}{'Optimal Weight (wᵢ)':>25}")
print("-"*75)
for i, c in enumerate(companies):
    print(f"{c:<15}{w_minvar[i]:>25.6f}")
print("-"*75)
print(f"{'Expected Return (μ₀)':<25}: {mu_min:.6f}")
print(f"{'Portfolio Std Dev (σ₀)':<25}: {std_min:.6f}")
print("="*75)

# ------------------------------------------------------------
# 7️⃣ PLOT EFFICIENT FRONTIER
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(std_v, mu_v, s=15, alpha=0.5, label="Random Portfolios")
plt.scatter(sigma, mu, s=100, color="black", marker="*", label="Individual Assets")
plt.scatter(std_min, mu_min, s=130, color="red", marker="X", label="S₀ (Min Var Portfolio)")

plt.title("Efficient Frontier (4-Asset Portfolio)")
plt.xlabel("Portfolio Std Dev (σv)")
plt.ylabel("Expected Return (μv)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

#code2
# =============================================================
# 📈 Modern Portfolio Theory — Multiple Securities + CML
# =============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# =============================================================
# 1️⃣ Load all stock CSVs dynamically
# =============================================================
file_paths = glob.glob("/content/*.csv")  # adjust your folder path
print(f"Loaded {len(file_paths)} stock files:")
for path in file_paths:
    print(" -", path.split("/")[-1])

# =============================================================
# 2️⃣ Compute daily returns (%)
# =============================================================
def calculate_returns(df):
    return ((df['Close Price'] - df['Open Price']) / df['Open Price']) * 100

returns = []
stock_names = []

for file in file_paths:
    df = pd.read_csv(file)
    stock_name = file.split("/")[-1].split(".")[0]
    stock_names.append(stock_name)
    returns.append(calculate_returns(df))

# Combine all returns into one DataFrame
returns_df = pd.concat(returns, axis=1)
returns_df.columns = stock_names
returns_df.dropna(inplace=True)

# =============================================================
# 3️⃣ Compute mean returns and covariance matrix
# =============================================================
mean_returns = returns_df.mean().values
cov_matrix = returns_df.cov().values
n = len(stock_names)
U = np.ones(n)

# =============================================================
# 4️⃣ Compute Minimum Variance Portfolio (MVP)
# =============================================================
inv_cov = np.linalg.inv(cov_matrix)
w_mvp = np.dot(inv_cov, U) / np.dot(U.T, np.dot(inv_cov, U))

mvp_return = np.dot(w_mvp, mean_returns)
mvp_risk = np.sqrt(np.dot(w_mvp.T, np.dot(cov_matrix, w_mvp)))

print("\n✅ Minimum Variance Portfolio Results")
print("Stocks:", stock_names)
print("Weights (%):", np.round(w_mvp * 100, 2))
print("Expected Return:", round(mvp_return, 4))
print("Expected Risk (Std Dev):", round(mvp_risk, 4))

# =============================================================
# 5️⃣ Analytical Efficient Frontier (Hyperbola)
# =============================================================
A = np.dot(mean_returns.T, np.dot(inv_cov, U))
B = np.dot(mean_returns.T, np.dot(inv_cov, mean_returns))
C = np.dot(U.T, np.dot(inv_cov, U))
D = B * C - A**2

target_returns = np.linspace(min(mean_returns), max(mean_returns), 200)
target_risks = np.sqrt((C * (target_returns**2) - 2 * A * target_returns + B) / D)

# =============================================================
# 6️⃣ Tangency Portfolio (Maximum Sharpe Ratio)
# =============================================================
risk_free_rate = 0.05  # <-- change this to your desired risk-free rate (% per period)

excess_returns = mean_returns - risk_free_rate
w_tan = np.dot(inv_cov, excess_returns) / np.dot(U.T, np.dot(inv_cov, excess_returns))
w_tan = w_tan / np.sum(w_tan)  # normalize to sum to 1

tan_return = np.dot(w_tan, mean_returns)
tan_risk = np.sqrt(np.dot(w_tan.T, np.dot(cov_matrix, w_tan)))
sharpe_ratio = (tan_return - risk_free_rate) / tan_risk

print("\n✅ Tangency Portfolio Results")
print("Weights (%):", np.round(w_tan * 100, 2))
print("Expected Return:", round(tan_return, 4))
print("Expected Risk (Std Dev):", round(tan_risk, 4))
print("Sharpe Ratio:", round(sharpe_ratio, 4))

# =============================================================
# 7️⃣ Random Portfolios (for visualization)
# =============================================================
num_portfolios = 5000
random_weights = np.random.dirichlet(np.ones(n), num_portfolios)
random_returns = random_weights.dot(mean_returns)
random_risks = np.sqrt(np.einsum('ij,jk,ik->i', random_weights, cov_matrix, random_weights))

# =============================================================
# 8️⃣ Capital Market Line (CML)
# =============================================================
cml_x = np.linspace(0, max(random_risks), 100)
cml_y = risk_free_rate + sharpe_ratio * cml_x

# =============================================================
# 9️⃣ Plot: Efficient Frontier + MVP + Tangency + CML
# =============================================================
plt.figure(figsize=(10, 6))
plt.scatter(random_risks, random_returns, c='lightgray', s=10, label='Random Portfolios')
plt.plot(target_risks, target_returns, 'b-', linewidth=2, label='Efficient Frontier')
plt.scatter(mvp_risk, mvp_return, c='red', marker='*', s=200, label='Minimum Variance Portfolio')
plt.scatter(tan_risk, tan_return, c='green', marker='D', s=120, label='Tangency Portfolio')
plt.plot(cml_x, cml_y, 'orange', linestyle='--', linewidth=2, label='Capital Market Line (CML)')

plt.title("Efficient Frontier, MVP & Capital Market Line (CML)", fontsize=14)
plt.xlabel("Risk (Standard Deviation)")
plt.ylabel("Expected Return (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

____________________________________________________________________________________________________________________________________________________________

#Global Alignment

import seaborn as sns

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Needleman-Wunsch global alignment.
    Returns (aligned_seq1, aligned_seq2, score)
    """
    n, m = len(seq1), len(seq2)

    # ---- 1. Build scoring matrix ---------------------------------
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            left = dp[i][j-1] + gap
            up   = dp[i-1][j] + gap
            dp[i][j] = max(diag, left, up)

    # ---- 2. Traceback ---------------------------------------------
    align1, align2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1

    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))

    return align1, align2, dp[n][m], dp   # dp is returned for printing


# ------------------------------------------------------------------
def print_matrix(seq1, seq2, dp, gap=-1):
    """
    Print the DP matrix with seq1 on the *columns* and seq2 on the *rows*.
    """
    n, m = len(seq1), len(seq2)

    # Header (seq1 on top)
    header = "   " + " ".join(f"{c:>3}" for c in " " + seq1)
    print(header)

    # Rows (seq2 on the left)
    for i in range(n + 1):
        row_label = seq2[i-1] if i > 0 else " "
        row = f"{row_label} " + " ".join(f"{dp[i][j]:>3}" for j in range(m + 1))
        print(row)


seq1 = "TGGTG"      # <-- will be shown on *columns*
seq2 = "ATCGT"   # <-- will be shown on *rows*

a1, a2, score, dp = needleman_wunsch(seq1, seq2)

print("=== Needleman-Wunsch (seq1 = columns) ===")
print(f"Seq1 (cols): {seq1}")
print(f"Seq2 (rows): {seq2}\n")
print(f"Alignment score: {score}")
print(f"   {a1}")
print(f"   {a2}\n")

print("Scoring matrix:")
print_matrix(seq1,seq2,dp)

def plot_traceback(seq1, seq2, dp):
    n, m = len(seq1), len(seq2)
    path_i, path_j = [n], [m]

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (1 if seq1[i-1] == seq2[j-1] else -1):
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] - 2:
            i -= 1
        else:
            j -= 1
        path_i.append(i)
        path_j.append(j)

    plt.figure(figsize=(8, 6))
    sns.heatmap(dp, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=["-"] + list(seq1),
                yticklabels=["-"] + list(seq2))
    plt.plot([j + 0.5 for j in path_j[::-1]], [i + 0.5 for i in path_i[::-1]],
             color="red", linewidth=2, marker="o")
    plt.title("Traceback Path (Optimal Alignment)")
    plt.xlabel("Sequence 1")
    plt.ylabel("Sequence 2")
    plt.show()

plot_traceback(seq1, seq2, dp)


____________________________________________________________________________________________________________________________________________________________



# ==========================================================
# 🔹 MANUAL ARMA MODELING FROM SCRATCH (HDFC.csv + Z-TEST CURVE + EQUATION)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

# ==========================================================
# STEP 1: Load and Stationarize the data
# ==========================================================

path = "/content/HDFC.csv"
df = pd.read_csv(path)

# Try to identify price column automatically
possible_cols = ['Close Price', 'Open Price', 'Adj Close', 'Last']
col = next((c for c in possible_cols if c in df.columns), df.columns[-1])
prices = df[col].dropna().values

# Convert to log returns
y = np.diff(np.log(prices))
y = y - np.mean(y)

print(f"Loaded {len(y)} log returns from {col}")
print(f"Mean ≈ {np.mean(y):.6f}, Std ≈ {np.std(y):.6f}")

# ==========================================================
# STEP 2: Define ACF and PACF
# ==========================================================

def acf(data, lag):
    n = len(data)
    c_k = []
    Y = np.mean(data)
    for k in range(lag + 1):
        sum_product = 0
        for t in range(n - k):
            p1 = data[t] - Y
            p2 = data[t + k] - Y
            sum_product += p1 * p2
        c_k.append(sum_product / n)
    P_k = np.array(c_k) / c_k[0] if c_k[0] != 0 else np.zeros(len(c_k))
    return P_k

def calculate_pacf(y, lags):
    rho = acf(y, lags)
    pacf_vals = [1.0]
    for k in range(1, lags + 1):
        P_k = np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k + 1])
        phi_k = np.linalg.solve(P_k, rho_k)
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)

# ==========================================================
# STEP 3: Identify AR(p) and MA(q) using ACF/PACF
# ==========================================================

lag = 20
rho = acf(y, lag)
pac = calculate_pacf(y, lag)
band = 1.96 / np.sqrt(len(y))

plt.figure(figsize=(10,4))
plt.subplot(121); plt.stem(rho); plt.title("ACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.subplot(122); plt.stem(pac); plt.title("PACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.tight_layout(); plt.show()

def cutoff(series):
    vals = np.abs(series[1:])
    above = vals > band
    for i in range(1, len(above)):
        if not above[i] and not above[i-1]:
            return i
    return 1

p = cutoff(pac)
q = cutoff(rho)
print(f"Suggested p={p}, q={q}")

# ==========================================================
# STEP 4: Define ARMA model
# ==========================================================

def arma_model(params, y, p, q):
    c = params[0]
    phi = params[1:p + 1]
    theta = params[p + 1:p + 1 + q]
    n = len(y)
    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_term = np.dot(phi, y[t - p:t][::-1]) if p > 0 else 0
        ma_term = np.dot(theta, eps[t - q:t][::-1]) if q > 0 else 0
        eps[t] = y[t] - (c + ar_term + ma_term)
    return eps

# ==========================================================
# STEP 5: Fit ARMA manually
# ==========================================================

def fit_arma_manual(y, p, q):
    init_params = np.zeros(1 + p + q)
    def objective(params):
        eps = arma_model(params, y, p, q)
        return np.sum(eps ** 2)
    res = minimize(objective, init_params, method='BFGS')
    fitted_params = res.x
    residuals = arma_model(fitted_params, y, p, q)
    return fitted_params, residuals

params, eps = fit_arma_manual(y, p, q)
print("\nFitted parameters:", params)

# --- Print ARMA equation ---
def print_arma_equation(params, p, q):
    eq = f"y_t = {params[0]:.4f}"
    for i in range(p):
        eq += f" + ({params[i+1]:.4f})·y_(t-{i+1})"
    for j in range(q):
        eq += f" + ({params[p+1+j]:.4f})·ε_(t-{j+1})"
    eq += " + ε_t"
    print("\n📘 Estimated ARMA Equation:")
    print(eq)

print_arma_equation(params, p, q)

# ==========================================================
# STEP 6: Residual Analysis + Z-Test Curve
# ==========================================================

res_acf = acf(eps[max(p, q):], 20)
mu, sigma = np.mean(eps), np.std(eps)

plt.figure(figsize=(14,5))
plt.subplot(1,3,1)
plt.plot(eps, label='Residuals'); plt.title("Residuals over time"); plt.legend()

plt.subplot(1,3,2)
plt.hist(eps, bins=25, density=True, alpha=0.6, label='Residuals')
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 200)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', lw=2, label='Normal Curve')
plt.legend(); plt.title("Residuals Histogram + Normal Fit")

plt.subplot(1,3,3)
plt.stem(res_acf)
plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.title("Residual ACF (White Noise Test)")
plt.tight_layout(); plt.show()

# Z-test visualization
z_scores = (eps - mu) / sigma
x = np.linspace(-4, 4, 200)
plt.figure(figsize=(7,4))
plt.hist(z_scores, bins=20, density=True, alpha=0.6, label='Residual z-scores')
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
plt.title("Z-Test Curve (Residuals vs Normal)")
plt.legend(); plt.show()

z_stat = (np.mean(eps)) / (np.std(eps) / np.sqrt(len(eps)))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"\nZ-test for residual mean=0 → z={z_stat:.3f}, p={p_value:.3f}")

____________________________________________________________________________________________________________________________________________________________

# ==========================================================
# 🔹 MANUAL ARIMA MODELING FROM SCRATCH (HDFC.csv + Z-TEST CURVE + EQUATION)
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats

path = "/content/HDFC.csv"
df = pd.read_csv(path)
possible_cols = ['Close Price', 'Open Price', 'Adj Close', 'Last']
col = next((c for c in possible_cols if c in df.columns), df.columns[-1])
prices = df[col].dropna().values

d = 1
y = np.log(prices)
y_diff = np.diff(y, n=d)
y_diff = y_diff - np.mean(y_diff)

print(f"Loaded {len(y_diff)} differenced returns from {col}")
print(f"ARIMA differencing order d={d}")

# --- ACF & PACF ---
def acf(data, lag):
    n = len(data)
    c_k = []
    Y = np.mean(data)
    for k in range(lag + 1):
        sum_product = 0
        for t in range(n - k):
            sum_product += (data[t] - Y) * (data[t + k] - Y)
        c_k.append(sum_product / n)
    P_k = np.array(c_k) / c_k[0]
    return P_k

def calculate_pacf(y, lags):
    rho = acf(y, lags)
    pacf_vals = [1.0]
    for k in range(1, lags + 1):
        P_k = np.array([[rho[abs(i - j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k + 1])
        phi_k = np.linalg.solve(P_k, rho_k)
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)

lag = 20
rho = acf(y_diff, lag)
pac = calculate_pacf(y_diff, lag)
band = 1.96 / np.sqrt(len(y_diff))

plt.figure(figsize=(10,4))
plt.subplot(121); plt.stem(rho); plt.title("ACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.subplot(122); plt.stem(pac); plt.title("PACF"); plt.axhline(band, color='red', linestyle='--'); plt.axhline(-band, color='red', linestyle='--')
plt.tight_layout(); plt.show()

def cutoff(series):
    vals = np.abs(series[1:])
    above = vals > band
    for i in range(1, len(above)):
        if not above[i] and not above[i-1]:
            return i
    return 1

p = cutoff(pac)
q = cutoff(rho)
print(f"Suggested p={p}, q={q}")

# --- ARIMA Model ---
def arma_model(params, y, p, q):
    c = params[0]
    phi = params[1:p + 1]
    theta = params[p + 1:p + 1 + q]
    n = len(y)
    eps = np.zeros(n)
    for t in range(max(p, q), n):
        ar_term = np.dot(phi, y[t - p:t][::-1]) if p > 0 else 0
        ma_term = np.dot(theta, eps[t - q:t][::-1]) if q > 0 else 0
        eps[t] = y[t] - (c + ar_term + ma_term)
    return eps

def fit_arima_manual(y, p, d, q):
    y_diff = np.diff(y, n=d) if d > 0 else y
    init_params = np.zeros(1 + p + q)
    def objective(params):
        eps = arma_model(params, y_diff, p, q)
        return np.sum(eps ** 2)
    res = minimize(objective, init_params, method='BFGS')
    fitted_params = res.x
    residuals = arma_model(fitted_params, y_diff, p, q)
    return fitted_params, residuals, y_diff

params, eps, y_diff = fit_arima_manual(y, p, d, q)
print("\nFitted parameters:", params)

# --- Print ARIMA Equation ---
def print_arima_equation(params, p, d, q):
    eq = f"(1 - B)^{d} y_t = {params[0]:.4f}"
    for i in range(p):
        eq += f" + ({params[i+1]:.4f})·y_(t-{i+1})"
    for j in range(q):
        eq += f" + ({params[p+1+j]:.4f})·ε_(t-{j+1})"
    eq += " + ε_t"
    print("\n📘 Estimated ARIMA Equation:")
    print(eq)

print_arima_equation(params, p, d, q)

# --- Residual Z-Test and Visualization ---
mu, sigma = np.mean(eps), np.std(eps)
z_scores = (eps - mu) / sigma
x = np.linspace(-4, 4, 200)
plt.hist(z_scores, bins=20, density=True, alpha=0.6, label='Residual z-scores')
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='N(0,1)')
plt.legend(); plt.title("Z-Test Curve (Residuals vs Normal)")
plt.show()

z_stat = (np.mean(eps)) / (np.std(eps) / np.sqrt(len(eps)))
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
print(f"Z-test for residual mean=0 → z={z_stat:.3f}, p={p_value:.3f}")

____________________________________________________________________________________________________________________________________________________________

# QUESTION 3
# Market vs Stock

import numpy as np

# Example returns (% converted to decimals)
R_xyz = np.array([0.02, 0.05, -0.01, 0.04, 0.03])
R_mkt = np.array([0.015, 0.04, -0.005, 0.035, 0.025])
prob = np.array([0.3, 0.25, 0.2, 0.2, 0.05])
Rf = 0.01  # 1% risk-free rate

T = len(R_xyz)

# mean
mu_xyz = np.sum(prob * R_xyz)
mu_mkt = np.sum(prob * R_mkt)

# Variances
sigma2_xyz = np.sum(prob * (R_xyz - mu_xyz) ** 2)
sigma2_mkt = np.sum(prob * (R_mkt - mu_mkt) ** 2)

# Covariance
cov_im = np.sum(prob * (R_xyz - mu_xyz) * (R_mkt - mu_mkt))

# Beta
beta_xyz = cov_im / sigma2_mkt

# Fair expected return (CAPM)
mu_v = Rf + beta_xyz * (mu_mkt - Rf)

print(f"Expected Return (μ_xyz): {mu_xyz:.4f}")
print(f"Market Return (μ_m): {mu_mkt:.4f}")
print(f"Variance of XYZ (σ²_xyz): {sigma2_xyz:.6f}")
print(f"Beta (β): {beta_xyz:.4f}")
print(f"Fair Return (μv): {mu_v:.4f}")

if mu_xyz > mu_v:
    print("Investment is GOOD (undervalued).")
else:
    print("Investment is NOT good (overvalued).")


#finance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1️⃣ LOAD CSV DATA
# ------------------------------------------------------------
companies = ["532540", "507685", "500209"]
N_sim = 5000

open_prices = pd.DataFrame()
close_prices = pd.DataFrame()

for c in companies:
    df = pd.read_csv(f"lab_test_final/{c}.csv", parse_dates=["Month"])
    df.set_index("Month", inplace=True)
    open_prices[c] = df["Open Price"]
    close_prices[c] = df["Close Price"]

# ------------------------------------------------------------
# 2️⃣ CALCULATE MONTHLY RETURNS (k1, k2, k3)
# ------------------------------------------------------------
returns = (close_prices - open_prices) / open_prices
returns = returns.dropna()
returns.columns = ["k1", "k2", "k3"]

# ------------------------------------------------------------
# 3️⃣ MEAN (μ) AND STANDARD DEVIATION (σ)
# ------------------------------------------------------------
mu = returns.mean().values
cov_matrix = returns.cov().values
sigma = np.sqrt(np.diag(cov_matrix))

mu1, mu2, mu3 = mu
std1, std2, std3 = sigma

print("\n" + "="*75)
print("📈  ASSET-WISE STATISTICS")
print("="*75)
print(f"{'Asset':<15}{'Mean (μ)':>15}{'Std Dev (σ)':>20}")
print("-"*75)
for i, c in enumerate(companies):
    print(f"{c:<15}{mu[i]:>15.6f}{sigma[i]:>20.6f}")
print("="*75)

print("\nCovariance Matrix (Σ):")
print("-"*75)
print(pd.DataFrame(np.round(cov_matrix, 6), index=companies, columns=companies))
print("\nCorrelation Matrix (ρ):")
print("-"*75)
print(pd.DataFrame(np.round(np.corrcoef(returns.T), 4), index=companies, columns=companies))
print("="*75)

# ------------------------------------------------------------
# 4️⃣ RANDOM WEIGHTS (SUM TO 1)
# ------------------------------------------------------------
def random_weights(n_assets, n_portfolios):
    w = np.random.rand(n_portfolios, n_assets)
    w = w / w.sum(axis=1)[:, None]
    return w


weights = random_weights(len(companies), N_sim)
print(weights)

# ------------------------------------------------------------
# 5️⃣ PORTFOLIO MEAN (μv) AND VARIANCE (σv²)
# ------------------------------------------------------------
mu_v = weights @ mu
var_v = np.array([w @ cov_matrix @ w.T for w in weights])
std_v = np.sqrt(var_v)

# ------------------------------------------------------------
# 6️⃣ CLOSED-FORM MINIMUM VARIANCE PORTFOLIO (S₀)
# ------------------------------------------------------------
ones = np.ones(len(companies)).T
inv_cov = np.linalg.inv(cov_matrix)
w_minvar = (ones @ inv_cov) / (ones @ inv_cov @ ones.T)
w_minvar = w_minvar.flatten()

mu_min = float(w_minvar @ mu)
var_min = float(w_minvar @ cov_matrix @ w_minvar.T)
std_min = np.sqrt(var_min)

print("\n\n" + "="*75)
print("🎯  CLOSED-FORM MINIMUM VARIANCE PORTFOLIO  (S₀)")
print("="*75)
print(f"{'Asset':<15}{'Optimal Weight (wᵢ)':>25}")
print("-"*75)
for i, c in enumerate(companies):
    print(f"{c:<15}{w_minvar[i]:>25.6f}")
print("-"*75)
print(f"{'Expected Return (μ₀)':<25}: {mu_min:.6f}")
print(f"{'Portfolio Std Dev (σ₀)':<25}: {std_min:.6f}")
print("="*75)

# ------------------------------------------------------------
# 7️⃣ PLOT EFFICIENT FRONTIER
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(std_v, mu_v, s=15, alpha=0.5, label="Random Portfolios")
plt.scatter(sigma, mu, s=100, color="black", marker="*", label="Individual Assets")
plt.scatter(std_min, mu_min, s=130, color="red", marker="X", label="S₀ (Min Var Portfolio)")

plt.title("Efficient Frontier (4-Asset Portfolio)")
plt.xlabel("Portfolio Std Dev (σv)")
plt.ylabel("Expected Return (μv)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()