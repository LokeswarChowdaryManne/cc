#Q1. Analyze the dynamical system

import random
import numpy as np
import matplotlib.pyplot as plt

# Generate a0 values
a0_values = [random.uniform(0.5, 1) for _ in range(10)]

# Generate r values
r1_values = [0] * 100
r2_values = [random.uniform(0, 1) for _ in range(100)]
r3_values = [random.uniform(-1, 0) for _ in range(100)]
r4_values = [random.choice([1, -1]) * random.uniform(1, 1000) for _ in range(100)]

labels = ['i. r=0', 'ii. 0<r<1', 'iii. -1<r<0', 'iv. r>1']

# i. r = 0
data = []
for a in a0_values:
    sequences = []
    for r in r1_values:
        sequence = []
        for n in range(100):
            sequence.append((r ** n) * a)
        sequences.append(sequence)
    data.append(sequences)

for sequences_for_a0 in data:
    for sequence in sequences_for_a0:
        plt.plot(np.array(sequence))
plt.title(labels[0])
plt.show()

# ii. 0 < r < 1
data = []
for a in a0_values:
    sequences = []
    for r in r2_values:
        sequence = []
        for n in range(100):
            sequence.append((r ** n) * a)
        sequences.append(sequence)
    data.append(sequences)

for sequences_for_a0 in data:
    for sequence in sequences_for_a0:
        plt.plot(np.array(sequence))
plt.title(labels[1])
plt.show()

# iii. -1 < r < 0
data = []
for a in a0_values:
    sequences = []
    for r in r3_values:
        sequence = []
        for n in range(100):
            sequence.append((r ** n) * a)
        sequences.append(sequence)
    data.append(sequences)

for sequences_for_a0 in data:
    for sequence in sequences_for_a0:
        plt.plot(np.array(sequence))
plt.title(labels[2])
plt.show()

# iv. r > 1 or r < -1
data = []
for a in a0_values:
    sequences = []
    for r in r4_values:
        sequence = []
        for n in range(100):
            sequence.append((r ** n) * a)
        sequences.append(sequence)
    data.append(sequences)

for sequences_for_a0 in data:
    for sequence in sequences_for_a0:
        plt.plot(np.array(sequence))
plt.title(labels[3])
plt.show()

#---------------------------------------------------------------------------------------------------------

'''
    Q2. 2.Construct a dynamical system for the following : Consider the decay of digoxin in the blood stream to precribed
    dosage that keeps the concentration between the acceptable levels. Suppose we prescribe a daily dosage of

    (i) 0.1 mg

    (ii) 0.2 mg

    (iii) 0.3 mg

    and know that half the digoxin remains in the system at the end of each dosage period.
    Plot he graphs and discuss the stability.
'''

import numpy as np
import matplotlib.pyplot as plt

def digoxin_decay(dose : float, days : int = 50) :
    conc = [0]

    for i in range(1, days) :
        conc.append(0.5 * conc[-1] + dose)
    return conc

colors = ['Red', 'Blue', 'Yellow']
days = 50
doses = [0.1, 0.2, 0.3]

plt.style.use("default")
for i, dose in enumerate(doses) :
    concn = digoxin_decay(dose, days)
    steady_state = 2 * dose

    plt.plot(range(days), concn, label = f"Dose = {dose} mg/day", color = colors[i])
    plt.axhline(steady_state, linestyle = '--', color = colors[i], alpha = 0.6)

plt.title("Digoxin Concentration Over Time")
plt.xlabel("Days")
plt.ylabel("Digoxin in Bloodstream (mg)")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#--------------------------------------------------------------------------------------------------------

'''
Q3. Generate
(i) 500,
(ii) 1000,
(iii) 10,000
(iv) 1,00,000
random numbers for the following distributions

Draw the histogram and frequency polygon
(i) Uniform distribution
(ii) exponential distribution
(iii) Weibull distribution
(iv) triangular distribution
'''

import numpy as np
import matplotlib.pyplot as plt

sample_sizes = [500, 1000, 10000, 100000]
distributions = ['uniform', 'exponential', 'weibull', 'triangular']

distribution_colors = {
    'uniform': ('red', 'blue'),
    'exponential': ('lightgreen', 'orange'),
    'weibull': ('green', 'red'),
    'triangular': ('pink', 'purple')
}

random_data = {}

for size in sample_sizes:
    random_data[size] = {}
    for dist in distributions:
        if dist == 'uniform':
            data = np.random.uniform(low=0, high=1, size=size)

        elif dist == 'exponential':
            data = np.random.exponential(scale=1.0, size=size)

        elif dist == 'triangular':
            data = np.random.triangular(left=0, mode=0.5, right=1.0, size=size)

        elif dist == 'weibull':
            data = np.random.weibull(a=2.0, size=size)

        random_data[size][dist] = data


for size, distributions_data in random_data.items():
    for dist, data in distributions_data.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        n_bins = 50
        hist_color, line_color = distribution_colors[dist]

        counts, bins, patches = ax.hist(
            data,
            bins=n_bins,
            density=True,
            alpha=0.7,
            label='Histogram',
            color=hist_color
        )

        bin_midpoints = (bins[:-1] + bins[1:]) / 2
        normalized_counts = counts

        ax.plot(
            bin_midpoints,
            normalized_counts,
            marker='o',
            linestyle='-',
            color=line_color,
            label='Frequency Polygon'
        )

        ax.set_title(f"Histogram and Frequency Polygon for {dist.capitalize()} Distribution (n={size})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()

        plt.tight_layout()
        plt.show()


#-------------------------------------------------------------------------------------------------------

'''
    Q4. Obtain the value of pie using Monte carlo simulation method by generating random numbers by between [0,1]  using the random number generated in problem (1)
'''

r2_values = [random.uniform(0, 1) for _ in range(100)]

if len(r2_values) % 2 != 0:
    r2_values.append(random.uniform(0, 1))

points = [(r2_values[i], r2_values[i + 1]) for i in range(0, len(r2_values), 2)]

# no of points fall inside the quarter circle
inside_circle = 0
for x, y in points:
    if x ** 2 + y ** 2 <= 1:
        inside_circle += 1

total_points = len(points)
pi_estimate = 4 * (inside_circle / total_points)

print(f"Estimated value of π using Monte Carlo method: {pi_estimate}")

#------------------------------------------------------------------------------------------------------------

#Q5

import numpy as np
import pandas as pd

PURCHASE_COST = 0.30
SELL_PRICE = 0.45
SCRAP_VALUE = 0.05
DAILY_SUPPLY = 70
SIM_DAYS = [200, 500, 1000, 10000]

NEWS_TYPE_RANGES = {
    "Good": (1, 35),
    "Fair": (36, 80),
    "Poor": (81, 100)
}

DEMAND_LOOKUP = {
    "Good": [(40, (1, 3)), (50, (4, 8)), (60, (9, 23)), (70, (24, 43)), (80, (44, 78)), (90, (79, 93)), (100, (94, 100))],
    "Fair": [(40, (1, 10)), (50, (11, 28)), (60, (29, 68)), (70, (69, 88)), (80, (89, 96)), (90, (97, 100))],
    "Poor": [(40, (1, 44)), (50, (45, 66)), (60, (67, 82)), (70, (83, 94)), (80, (95, 100))]
}


def determine_news_type(rand_num):
    for category, (low, high) in NEWS_TYPE_RANGES.items():
        if low <= rand_num <= high:
            return category
    return "Unknown"


def generate_demand_number(news_type):
    if news_type == "Good":
        return int(np.clip(np.random.exponential(50), 1, 100))
    elif news_type == "Fair":
        return int(np.clip(np.random.normal(50, 10), 1, 100))
    else:
        return int(np.clip(np.random.poisson(50), 1, 100))


def map_to_demand(news_type, rand_val):
    for demand_val, (low, high) in DEMAND_LOOKUP[news_type]:
        if low <= rand_val <= high:
            return demand_val
    return 0


def calculate_metrics(demand):
    sold = min(demand, DAILY_SUPPLY)
    unsold = DAILY_SUPPLY - sold
    lost_sales = max(0, demand - DAILY_SUPPLY)

    revenue = sold * SELL_PRICE
    salvage = unsold * SCRAP_VALUE
    cost = DAILY_SUPPLY * PURCHASE_COST
    lost_profit = lost_sales * (SELL_PRICE - PURCHASE_COST)

    profit = revenue + salvage - cost
    return sold, revenue, lost_profit, salvage, profit


def simulate_news_sales(n_days):
    records = []

    for day in range(1, n_days + 1):
        news_rand = np.random.randint(1, 101)
        news_type = determine_news_type(news_rand)

        demand_rand = generate_demand_number(news_type)
        demand = map_to_demand(news_type, demand_rand)

        sold, revenue, loss, salvage, profit = calculate_metrics(demand)

        records.append({
            "Day": day,
            "NewsType": news_type,
            "NewsRand": news_rand,
            "DemandRand": demand_rand,
            "Demand": demand,
            "Sold": sold,
            "Revenue": round(revenue, 2),
            "LostProfit": round(loss, 2),
            "Salvage": round(salvage, 2),
            "Profit": round(profit, 2)
        })

    return pd.DataFrame(records)


def run_all_simulations():
    all_results = {}
    for days in SIM_DAYS:
        df = simulate_news_sales(days)
        all_results[days] = df
    return all_results



simulations = run_all_simulations()

for days, df in simulations.items():
    print(f"\nSimulation Results for {days} Days (First 5 Days):\n")
    print(df.head(5).to_string(index=False))
    print(f"\nSummary for {days} Days:")
    print(df[["Revenue", "LostProfit", "Salvage", "Profit"]].sum())


#------------------------------------------------------------------------------------------------------

'''
    Q6. The inter arrival time of the customer at a booking station is exponential with men 10 min and the
    service time for booking follows Poisson distribution between 8 and 12 min. The customers are
    served FIFO basis. Simulate the system for 1000 time units . From the results of the simulation
    determine the average time customer waits in a queue, average number of customers waiting
    and average utilization of the booking station Plot the sample path respect to the time.
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
SIM_TIME = 1000
MEAN_INTER_ARRIVAL = 10
SERVICE_POISSON_MEAN = 10
MIN_SERVICE_TIME = 8
MAX_SERVICE_TIME = 12


arrival_times = []
service_start_times = []
departure_times = []
queue_lengths = []
time_points = []

current_time = 0
next_arrival_time = np.random.exponential(MEAN_INTER_ARRIVAL)
next_departure_time = float('inf')

server_busy = False
queue = []


while current_time < SIM_TIME:
    if next_arrival_time < next_departure_time:
        current_time = next_arrival_time
        arrival_times.append(current_time)

        if server_busy:
            queue.append(current_time)
        else:
            service_time = np.clip(np.random.poisson(SERVICE_POISSON_MEAN), MIN_SERVICE_TIME, MAX_SERVICE_TIME)
            service_start_times.append(current_time)

            departure_time = current_time + service_time
            departure_times.append(departure_time)

            next_departure_time = departure_time
            server_busy = True

        next_arrival_time += np.random.exponential(MEAN_INTER_ARRIVAL)
    else:
        current_time = next_departure_time

        if queue:
            arrival_from_queue = queue.pop(0)
            service_time = np.clip(np.random.poisson(SERVICE_POISSON_MEAN), MIN_SERVICE_TIME, MAX_SERVICE_TIME)
            service_start_times.append(current_time)

            departure_time = current_time + service_time
            departure_times.append(departure_time)

            next_departure_time = departure_time
        else:
            server_busy = False
            next_departure_time = float('inf')

    queue_lengths.append(len(queue))
    time_points.append(current_time)


num_customers = len(arrival_times)

wait_times = [start - arrival for start, arrival in zip(service_start_times, arrival_times)]
avg_wait_time = np.mean(wait_times)

avg_queue_length = np.mean(queue_lengths)
utilization = sum([d - s for s, d in zip(service_start_times, departure_times)]) / SIM_TIME
print("Total customers:", num_customers)
print("Average waiting time in queue: {:.2f} min".format(avg_wait_time))
print("Average number of customers in queue: {:.2f}".format(avg_queue_length))
print("Booking station utilization: {:.2f}%".format(utilization * 100))

plt.figure(figsize=(10, 5))
plt.plot(time_points, queue_lengths, drawstyle='steps-post', label='Queue Length')
plt.xlabel("Time")
plt.ylabel("Number of Customers in Queue")
plt.title("Queue Length Over Time (Sample Path)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

