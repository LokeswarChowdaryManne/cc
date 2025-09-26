import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def second_order_exponential_smoothing_y1(data, alpha):
    first_order = [data[0]]
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * first_order[-1]
        first_order.append(smoothed_value)

    second_order = [first_order[0]]
    for i in range(1, len(first_order)):
        smoothed_value = alpha * first_order[i] + (1 - alpha) * second_order[-1]
        second_order.append(smoothed_value)

    res = [2*first_order[i] - second_order[i] for i in range(len(data))]
    return np.array(first_order), np.array(second_order), np.array(res)

def second_order_exponential_smoothing_mean(data, alpha):
    ymean = np.mean(data)

    first_order = [ymean]
    for i in range(1, len(data)):
        smoothed_value = alpha * data[i] + (1 - alpha) * first_order[-1]
        first_order.append(smoothed_value)

    second_order = [first_order[0]]
    for i in range(1, len(first_order)):
        smoothed_value = alpha * first_order[i] + (1 - alpha) * second_order[-1]
        second_order.append(smoothed_value)

    res = [2*first_order[i] - second_order[i] for i in range(len(data))]
    return np.array(first_order), np.array(second_order), np.array(res)

def run_ttest(before, after, label):
    d = before - after
    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)
    n = len(d)
    t_stat = d_mean / (d_std / np.sqrt(n))
    p_val = 2 * t.sf(np.abs(t_stat), df=n-1)
    print(f"{label}: t={t_stat:.4f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("  Reject H0: significant difference\n")
    else:
        print("  Fail to Reject H0: no significant difference\n")

data=np.array([43.1, 43.7, 45.3, 47.3, 50.6, 54.0, 46.2, 49.3, 53.9, 42.5,
               41.8, 50.7, 55.8, 48.7, 48.2, 46.9, 47.4, 49.2, 50.9, 55.3,
               47.7, 51.1, 67.1, 47.2, 50.4, 44.2, 52.0, 35.5, 48.4, 55.4,
               52.9, 47.3, 50.0, 56.7, 42.3, 52.0, 48.6, 51.5, 49.5, 51.4,
               48.3, 45.0, 55.2, 63.7, 64.4, 66.8, 63.3, 60.0, 60.9, 56.1])

alpha = 0.2

f1_y1, s1_y1, res1_y1 = second_order_exponential_smoothing_y1(data, alpha)
f1_mean, s1_mean, res1_mean = second_order_exponential_smoothing_mean(data, alpha)

plt.figure(figsize=(10,5))
plt.plot(data, label="Original", marker="o", color="black")

plt.plot(f1_y1, label="First Order (start=y1)", linestyle="--", marker="s")
plt.plot(f1_mean, label="First Order (start=mean)", linestyle="--", marker="^")

plt.plot(res1_y1, label="Second Order (start=y1)", linestyle="-.", marker="s")
plt.plot(res1_mean, label="Second Order (start=mean)", linestyle="-.", marker="^")

plt.legend()
plt.title("First and Second Order Exponential Smoothing")
plt.xlabel("Period")
plt.ylabel("Value")
plt.grid(True)
plt.show()

print("T-tests comparing Original vs Smoothed:\n")
run_ttest(data, f1_y1, "First Order (start=y1)")
run_ttest(data, f1_mean, "First Order (start=mean)")
run_ttest(data, res1_y1, "Second Order (start=y1)")
run_ttest(data, res1_mean, "Second Order (start=mean)")
