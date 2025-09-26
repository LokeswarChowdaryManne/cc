#REGRESSION

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

# Quadratic Regression

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

x_cubic = np.column_stack((np.ones(len(x)), x, x*2, x*3))

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

y_pred_cubic = intercept_cubic + slope_cubic_x * x + slope_cubic_x2 * (x*2) + slope_cubic_x3 * (x*3)

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

______________________________________________________________________________________________________________________________________________________________________________

#MOVING AVERAGE

# -- coding: utf-8 --
"""
Created on Wed Sep 24 19:20:05 2025

@author: shwet
"""
import matplotlib.pyplot as plt
import pandas as pd

#moving average
years=[1992,1993,1994,1995]
data = [293,246,231,282,301,252,227,291,304,259,239,296,306,265,240,300]
year=list(range(1,17))

#1 - year , 2-quarter , 3-data,  4-quarter moving total , 5- 4quarter moving total avg , 6 - 4quarter centered moving avg, 7 - ratio of centered to actual (6)/(3)

#linear regression
n=len(data)
sum_x=sum(year)
sum_y=sum(data)
sum_xy = sum([x*y for x,y in zip(year,data)])
sum_x2 = sum([x**2 for x in year])
#y=a+bx
b= ((n*sum_xy) - (sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
a = (sum_y - (b*sum_x))/n


y_pred = [a + b*x for x in year]
plt.scatter(year, data)
plt.plot(year,y_pred, "-",color="red")
plt.show()

###############################################
#rel cyc residual
#trend percentage


###############################################
#moving avg

moving_total=[]
moving_total_avg=[]

for i in range(len(data)-3):
    moving_total.append(sum(data[i:i+4]))
    moving_total_avg.append(sum(data[i:i+4])/4)
    
centered_moving_avg = []
for i in range(len(moving_total_avg)-1):
    centered_moving_avg.append(sum(moving_total_avg[i:i+2])/2)
    

actual_to_moving=[]
for i in range(len(centered_moving_avg)):
    val=(data[i+2]/centered_moving_avg[i])*100
    actual_to_moving.append(val)
    
df=pd.DataFrame(
    {
         "year": [1992]+[""]*3 + [1993]+[""]*3 + [1994]+[""]*3 + [1995]+[""]*3,
         "quarters": ["Winter","Spting","Summer","Fall"] * 4,
         "Data": data,
         "4 Quarter moving total" : [None]*2 + moving_total + [None],
         "4 Quarter moving avg": [None]*2 + moving_total_avg + [None],
         "4 Quarter centered moving avg" : [None]*2 + centered_moving_avg + [None]*2,
         "Percentage of actual to moving" :[None]*2 + actual_to_moving + [None]*2
     
     })

#print(df["year"],df["quarters"],df["Data"],df["4 Quarter moving total"], df["4 Quarter moving avg"],df["4 Quarter centered moving avg"], df["Percentage of actual to moving"] )

print(df.to_string(float_format="{:.2f}".format))
#################################################
#seasonal index
quarters={0:[],1:[],2:[],3:[]}
for i in range(len(actual_to_moving)):
    quarters[(i+2)%4].append(actual_to_moving[i])
    
modified_mean=[]
for q in quarters:
    mini=min(quarters[q])
    maxi=max(quarters[q])
    mean=(sum(quarters[q])-mini-maxi)/2
    print(f"Quarter {q+1}:",quarters[q])
    print(f"Min: {mini} , Max : {maxi}")
    print("MEAN:", mean)
    modified_mean.append(mean)
    
    
total_indices=sum(modified_mean)
if total_indices<400:
    adj_constant=400/total_indices
else:
    adj_constant=total_indices/400
    
seasonal_indice=[x*adj_constant for x in modified_mean]


#deseasonalized data
deseasonalized_data=[]
for i in range(0,len(data),4):
    deseasonalized_data.append(data[i]/(seasonal_indice[0]/100))
    deseasonalized_data.append(data[i+1]/(seasonal_indice[1]/100))
    deseasonalized_data.append(data[i+2]/(seasonal_indice[2]/100))
    deseasonalized_data.append(data[i+3]/(seasonal_indice[3]/100))


df2 = pd.DataFrame({
    "Year": [1992]+[""]*3 + [1993]+[""]*3 + [1994]+[""]*3 + [1995]+[""]*3,
    "Quarter": ["Winter","Spting","Summer","Fall"] * 4,
    "Given data": data,
    "Deseasonalized Data":deseasonalized_data
})
print(df2)


#do regression for deseasonalised data
n=len(deseasonalized_data)
sum_x=sum(year)
sum_y=sum(deseasonalized_data)
sum_xy = sum([x*y for x,y in zip(year,deseasonalized_data)])
sum_x2 = sum([x**2 for x in year])
#y=a+bx
b= ((n*sum_xy) - (sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
a = (sum_y - (b*sum_x))/n


y_pred = [a + b*x for x in year]
plt.scatter(year, data)
plt.plot(year,y_pred, "-",color="red")
plt.show()

#relative cyclic residual
rel_cyclic_resid = [((y_act - y_pre) / y_pre) * 100 for y_pre, y_act in zip(y_pred, data)]
print("Relative cyclic residual :")
for yr,rel in zip(year,rel_cyclic_resid ):
    print(f"{yr} : {rel:.2f}")

#plot original,deaseasonalized and the trend
plt.plot(year, data ,color="black",label="original")
plt.plot(year,deseasonalized_data,color="blue",label="De-seasonalized")
plt.plot(year, y_pred,color="red",label="trend(regression)")
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("Comparison plot")
plt.legend()
plt.show()

########################################################################3
#4

import matplotlib.pyplot as plt
import pandas as pd

years = [1992, 1993, 1994, 1995]
year=list(range(1,17))
#data is yearly spring summer fall winter
data=[293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
#print(year,len(year))
#print(data,len(data))

#fit regression line
sum_y = sum(data)
sum_x= sum(year)
sum_xy = sum([x*y for x,y in zip(year,data)])
sum_x2 = sum([x**2 for x in year])

# y = a + bx
a = ((sum_y * sum_x2) - (sum_x * sum_xy)) / ((len(year) * sum_x2) - sum_x**2)
b = ((len(year) * sum_xy )- (sum_x * sum_y)) / ((len(year) * sum_x2) - sum_x**2)

print(f"The equation that best fits the data is : y = {a:.2f} + {b:.2f}x")
y_pred=[a + b*x for x in year]

plt.scatter(year, data ,color="black")
plt.plot(year, y_pred,color="red")
plt.title(f"Linear Regression : y = {a:.2f} + {b:.2f}x ")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

#4 quarter moving total (4) & (5)
moving_total=[]
moving_total_average=[]
for i in range(len(data)-3):
    moving_total.append(sum(data[i:i+4]))
    moving_total_average.append(sum(data[i:i+4])/4)



#4 quartered centered moving avg (6)
centered_moving_average=[]
for i in range(len(moving_total_average)-1):
  centered_moving_average.append(sum(moving_total_average[i:i+2])/2)


#actual to moving percentage - (3)/(6) x 100
actual_to_moving=[]
for i in range(len(centered_moving_average)):
  actual_to_moving.append((data[i+2]/centered_moving_average[i])*100)

#print("Moving total:",moving_total)
#print("moving_total_average:",moving_total_average)
#print("centered_moving_average",centered_moving_average)
#print("actual_to_moving:",actual_to_moving)

"""display_year=["1991","","","","1992","","","" ,"1993","","","","1994","","","" ,"1995","","",""]
Quarter=["Spring","Summer","Fall","Winter"]*5
df=pd.DataFrame({
    "Year":display_year,
    "Quarter":Quarter,
    "Value (3)":data,
    "Moving Total (4)":moving_total,
    "Moving Total Average (5)":moving_total_average,
    "Centered Moving Average (6)":centered_moving_average,
    "Actual to Moving Percentage (3)/(6)*100":actual_to_moving
})"""

quarters= {0:[],1:[],2:[],3:[]}
for i in range(len(actual_to_moving)):
  quarters[(i+2)%4].append(actual_to_moving[i])
print(quarters)

#min and max val for each quarter
Modified_mean=[]
for q in quarters:
  min=quarters[q][0]
  max=quarters[q][0]
  for val in quarters[q]:
    if val<min:
      min=val
    if val>max:
      max=val
  print(f"Quarter {q+1}:")
  print("MIN : ", min)
  print("MAX : ", max)
  Modified_mean.append((sum(quarters[q])-(min+max))/2)
print(Modified_mean)

total_of_indices=sum(Modified_mean)
print(total_of_indices)

if total_of_indices<400:
  adjusting_constant=400/total_of_indices
else:
  adjusting_constant=total_of_indices/400
print("Adjusting constant:", adjusting_constant)


#seasonal indice
seasonal_indices=[x*adjusting_constant for x in Modified_mean]
print("Seasonal indices:",seasonal_indices)
print("Mean of seasonal indices:",sum(seasonal_indices)/4)

# Create DataFrame for display
display_year = ["1992","","","","1993","","","","1994","","","","1995","","",""]
Quarter = ["Spring","Summer","Fall","Winter"]*4
df = pd.DataFrame({
    "Year": display_year,
    "Quarter": Quarter,
    "Value (3)": data,
    "Moving Total (4)": [None]*3 + moving_total + [None]*0,  # Pad to length 16
    "Moving Total Average (5)": [None]*3 + moving_total_average + [None]*0,  # Pad to length 16
    "Centered Moving Average (6)": [None]*3 + centered_moving_average + [None]*1,  # Pad to length 16
    "Actual to Moving Percentage (3)/(6)*100": [None]*3 + actual_to_moving + [None]*1  # Pad to length 16
})

# Print DataFrame with formatted floating-point numbers
print(df.to_string(float_format="{:.2f}".format))

#plot centered moving average
plt.scatter(year, data ,color="black")
plt.plot(year[3:-1], centered_moving_average,color="red")
plt.title(f"Centered Moving average")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

deseasoned_data=[]
#Deseasonalising data
for i in range(0,len(data),4):
  deseasoned_data.append(data[i]/(seasonal_indices[0]/100))
  deseasoned_data.append(data[i+1]/(seasonal_indices[1]/100))
  deseasoned_data.append(data[i+2]/(seasonal_indices[2]/100))
  deseasoned_data.append(data[i+3]/(seasonal_indices[3]/100))

df2 = pd.DataFrame({
    "Year": display_year,
    "Quarter": Quarter,
    "Given data": data,
    "Deseasonalized Data":deseasoned_data
})

df2.head(10)

#fitting regression for deseasonalised data

sum_y = sum(deseasoned_data)
sum_x= sum(year)
sum_xy = sum([x*y for x,y in zip(year,deseasoned_data)])
sum_x2 = sum([x**2 for x in year])

# y = a + bx
a = ((sum_y * sum_x2) - (sum_x * sum_xy)) / ((len(year) * sum_x2) - sum_x**2)
b = ((len(year) * sum_xy )- (sum_x * sum_y)) / ((len(year) * sum_x2) - sum_x**2)

print(f"The equation that best fits the data is : y = {a:.2f} + {b:.2f}x")
y_pred=[a + b*x for x in year]

plt.scatter(year, data ,color="black")
plt.plot(year, y_pred,color="red")
plt.title(f"Linear Regression (Deseasonalized): y = {a:.2f} + {b:.2f}x ")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

#relative cyclic residual
rel_cyclic_resid = [((y_act - y_pre) / y_pre) * 100 for y_pre, y_act in zip(y_pred, data)]
print("Relative cyclic residual :")
for yr,rel in zip(year,rel_cyclic_resid ):
    print(f"{yr} : {rel:.2f}")

#plot original,deaseasonalized and the trend
plt.plot(year, data ,color="black",label="original")
plt.plot(year,deseasoned_data,color="blue",label="De-seasonalized")
plt.plot(year, y_pred,color="red",label="trend(regression)")
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("Comparison plot")
plt.legend()
plt.show()

______________________________________________________________________________________________________________________________________________________________________________

#EXPONENTIAL SECOND ORDER

# -- coding: utf-8 --
"""
Created on Thu Sep 25 15:02:13 2025

@author: shwet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# Data
yt = np.array([
    48.7, 45.8, 46.4, 46.2, 44.0,
    53.8, 47.6, 47.0, 47.6, 51.1,
    49.1, 46.7, 47.8, 45.8, 45.5,
    49.2, 54.8, 44.7, 51.1, 47.3,
    45.3, 43.3, 44.6, 47.1, 53.4,
    44.9, 50.5, 48.1, 45.4, 51.6,
    50.8, 46.4, 52.3, 50.5, 53.4,
    53.9, 52.3, 53.0, 48.6, 52.4,
    47.9, 49.5, 44.0, 53.8, 52.5,
    52.0, 50.6, 48.7, 51.4, 47.7
], dtype=float)

lam = 0.3
train_n = 35
y_train = yt[:train_n]
y_test = yt[train_n:]
periods = np.arange(1, len(yt) + 1)

# Second-order exponential smoothing on training set
n = len(y_train)
y1 = np.zeros(n)
y2 = np.zeros(n)
yhat = np.zeros(n)

y1[0] = y_train[0]
y2[0] = y1[0]
yhat[0] = y_train[0]

for t in range(1, n):
    y1[t] = lam * y_train[t] + (1 - lam) * y1[t - 1]
    y2[t] = lam * y1[t] + (1 - lam) * y2[t - 1]
    yhat[t] = 2 * y1[t] - y2[t]

# Compute a_T (level) and b_T (trend) at last training point
y1_T = y1[-1]
y2_T = y2[-1]

a_T = 2 * y1_T - y2_T
b_T = (lam / (1 - lam)) * (y1_T - y2_T)

# Forecast test set using (a_T + b_T * h)
h_steps = np.arange(1, len(y_test) + 1)
yhat_test = a_T + b_T * h_steps

pred_df = pd.DataFrame({
    "Period": periods[train_n:],
    "Actual": y_test,
    "Forecast": yhat_test,
    "Residual": y_test - yhat_test
})


smooth_df = pd.DataFrame({
    "Period": periods[:train_n],
    "y": y_train,
    "y(1)": y1,
    "y(2)": y2,
    "y_hat": yhat
})

print("\nSecond-Order Exponential Smoothing Table (Training Set):")
print(smooth_df.to_string(index=False))

# Manual one-sample t-test
res = pred_df["Residual"].values
n_test = len(res)
mean_res = np.mean(res)
sd_res = np.std(res, ddof=1)
se_res = sd_res / sqrt(n_test)
t_stat = mean_res / se_res if se_res > 0 else float('nan')

# Results
print("Level-Trend estimates at T=35:")
print(f"a_T (level) = {a_T:.4f}")
print(f"b_T (trend) = {b_T:.4f}")

print("\nManual t-test on residuals (H0: mean=0):")
print(f"t = {t_stat:.4f}")

print("\nForecast Table (Test Periods):")
print(pred_df.to_string(index=False))


# Plot
plt.figure(figsize=(10,4))
plt.plot(periods[:train_n], y_train, label="Train Actual", marker="o")
plt.plot(periods[:train_n], y1, label="First Smoother (ỹ(1))", linestyle="-")
plt.plot(periods[:train_n], y2, label="Second Smoother (ỹ(2))", linestyle="-.")
plt.plot(periods[:train_n], yhat, label="Train Fitted", linestyle="--")
plt.legend()
plt.xlabel("Period")
plt.ylabel("y")
plt.title("Second-Order Exponential Smoothing with Level & Trend (a_T, b_T)")
plt.show()

______________________________________________________________________________________________________________________________________________________________________________

#EXPONENTIAL FIRST ORDER

# -- coding: utf-8 --
"""
Created on Thu Sep 25 15:01:18 2025

@author: shwet
"""

#3rd question

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

periods = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
yt = np.array([np.random.uniform(0, 1) for _ in range(10)])

def exponential_smoothing_with_y1(yt, lambda_, y1):
    smoothed = np.zeros(len(yt))
    smoothed[0] = y1
    for t in range(1, len(yt)):
        smoothed[t] = lambda_ * yt[t] + (1 - lambda_) * smoothed[t - 1]
    return smoothed

def exponential_smoothing_with_ymean(yt, lambda_, ymean):
    smoothed = np.zeros(len(yt))
    smoothed[0] = ymean
    for t in range(1, len(yt)):
        smoothed[t] = lambda_ * yt[t] + (1 - lambda_) * smoothed[t - 1]
    return smoothed

for i in range(10):
  lambda_=  np.random.uniform(0, 1)
  smoothed_y1 = exponential_smoothing_with_y1(yt, lambda_, yt[0])
  smoothed_ymean = exponential_smoothing_with_ymean(yt, lambda_, np.mean(yt))

  data = {
      'Period': periods,
      'Original': yt,
      'Smoothed_y1': smoothed_y1,
      'Smoothed_ymean': smoothed_ymean

  }
  df = pd.DataFrame(data)
  print(df)


  plt.plot(periods, yt, label='Original Data', marker='o')
  plt.plot(periods, smoothed_y1, label=f'Smoothed at lambda={lambda_} and y = y1', marker='o')
  plt.plot(periods, smoothed_ymean, label=f'Smoothed at lambda={lambda_} and y = ymean', marker='^')
  plt.xlabel('Period')
  plt.ylabel('yt')
  plt.title('Original vs Smoothed Data')
  plt.legend()
  plt.grid(True)
  plt.xticks(periods)
  plt.tight_layout()
  plt.show()


  #ttest
  import numpy as np
  from scipy.stats import t
  before = yt
  after1=smoothed_y1
  after2=smoothed_ymean

  d1 = before- after1
  d2 = before - after2

  d_mean1 = np.mean(d1)
  d_mean2 = np.mean(d2)

  d_std1 = np.std(d1, ddof=1)
  d_std2 = np.std(d2, ddof=1)

  n = len(d1)

  t_stat1 = d_mean1 / (d_std1 / np.sqrt(n))
  t_stat2 = d_mean2 / (d_std2 / np.sqrt(n))

  p_val1 = 2 * t.sf(np.abs(t_stat1), df=n-1)
  p_val2 = 2 * t.sf(np.abs(t_stat2), df=n-1)


  alpha = 0.05
  print("t-statistic:", t_stat1)
  print("p-value:", p_val1)

  if p_val1 < alpha:
      print(" For y0^ = y1")
      print(f"Reject H0 at α={alpha}: Significant difference between before and after.")
  else:
      print(f"Fail to Reject H0 at α={alpha}: No significant difference between before and after.")

  print("t-statistic:", t_stat2)
  print("p-value:", p_val2)
  if p_val2 < alpha:
      print(" For y0^ = ymean")
      print(f"Reject H0 at α={alpha}: Significant difference between before and after.")
  else:
      print(f"Fail to Reject H0 at α={alpha}: No significant difference between before and after.")

______________________________________________________________________________________________________________________________________________________________________________

#ACF AND PACF

# -- coding: utf-8 --
"""
Created on Wed Sep 24 21:04:45 2025

@author: shwet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

y = np.array([29,20,25,29,31,33,34,27,26,30,
              29,28,28,26,27,26,30,28,26,30,
              31,30,37,30,33,31,27,33,37,29,
              28,30,29,34,30,20,17,23,24,34,
              36,35,33,29,25,27,30,29,28,32])

ymean=np.mean(y)
n=len(y)
lags=25

def acf(series,lags):
    n=len(series)
    ymean=np.mean(y)
    demeaned=y-ymean
    c0=np.sum( demeaned**2)/n
    acf_values=[1]
    for k in range(1,lags+1):
        ck=np.sum(demeaned[:-k]*demeaned[k:])/n
        acf_values.append(ck/c0)
    return acf_values

acf_k=acf(y, lags)

conf = 1.96/np.sqrt(n)

plt.stem(range(len(acf_k)),acf_k, basefmt=" ")
plt.axhline(0, color="black")
plt.axhline(conf, color="red",linestyle="--")
plt.axhline(- conf, color="red",linestyle="--")
plt.title("Sample ACF")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.show()
plt.show()


######################################################

def pacf(series,lags):
    rho=acf(series, lags)
    pacf_vals=[1]
    
    for k in range(1,lags+1):
        p_k=np.array([[rho[abs(i-j)] for j in range(k)] for i in range(k)])
        rho_k = np.array(rho[1:k+1])
        phi_k=np.linalg.solve(p_k,rho_k)
        pacf_vals.append(phi_k[-1])
    return np.array(pacf_vals)

pacf_k=pacf(y, lags)
    
conf = 2/np.sqrt(n)

plt.stem(range(len(pacf_k)),pacf_k, basefmt=" ")
plt.axhline(0, color="black")
plt.axhline(conf, color="red",linestyle="--")
plt.axhline(- conf, color="red",linestyle="--")
plt.title("Sample PACF")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.show()
plt.show()



print("PACF VALUES WITH SIGNIFICANCE TESTING")
print("=" * 65)
print(f"{'Lag':<6} {'PACF':<10} {'Significant?':<12} {'Decision':<40}")
print("-" * 65)

for lag, val in enumerate(pacf_k):
    if lag == 0:
        significant = "N/A"
        decision = "PACF(0) = 1 (by definition)"
    else:
        # Null hypothesis: PACF(lag) = 0
        # Reject H0 if |PACF| > 2/root(n)
        if abs(val) > conf:
            significant = "Yes"
            decision = "Reject H0: Significant partial autocorrelation"
        else:
            significant = "No"
            decision = "Fail to reject H0: Not significant"

    print(f"{lag:<6} {val:<10.4f} {significant:<12} {decision:<40}")

______________________________________________________________________________________________________________________________________________________________________________

#EASIER MOVING AVERAGE

“””Parallel Breakfast Foods has data on the number of boxes of cereal it has sold in each of the last 7 years.﻿﻿﻿﻿Find the linear estimating equation that best describes these data.
	•	﻿﻿﻿﻿Calculate the percent of trend for these data.
	•	﻿﻿﻿﻿Calculate the relative cyclical residual for these data.
	•	﻿﻿﻿﻿In which year does the biggest fluctuation from the trend occur under each measure of cyclical variation? Is this year the same for both measures? Explain.”””

#1.-----------------------------------------------------------------------

year=np.array(list(range(1989,1996)))
boxes=np.dot([21.0, 19.4, 22.6, 28.2, 30.4, 24.0, 25.0],10000.0)

mean_year = np.mean(year)
mean_boxes = np.mean(boxes)

numerator = np.sum((year - mean_year) * (boxes - mean_boxes))
denominator = np.sum((year - mean_year)**2)

slope = numerator / denominator
intercept = mean_boxes - slope * mean_year

print(f"The linear estimating equation (manual calculation) is: y = {slope:.2f}x + {intercept:.2f}")

import prettytable as pt

trend_values = slope * np.array(year) + intercept

percent_of_trend = (boxes / trend_values) * 100

relative_cyclical_residual = ((boxes - trend_values) / trend_values)*100

table = pt.PrettyTable()
table.field_names = ["Year", "Actual Boxes", "Trend Value", "Percent of Trend", "Relative Cyclical Residual"]

for i in range(len(year)):
    table.add_row([year[i], f"{boxes[i]:.2f}", f"{trend_values[i]:.2f}", f"{percent_of_trend[i]:.2f}%", f"{relative_cyclical_residual[i]:.2f}%"])

print(table)

# year with the largest absolute "Percent of Trend"
year_max_percent_trend = year[np.argmax(np.abs(percent_of_trend))]
year_max_trend_val=max(100-np.abs(percent_of_trend))
print(f"The year with the biggest fluctuation under 'Percent of Trend' is: {year_max_percent_trend}")
print(f"The biggest fluctuation under 'Percent of Trend' is: {year_max_trend_val}%")

# year with the largest absolute "Relative Cyclic Residuals"
year_max_relative_residual = year[np.argmax(np.abs(relative_cyclical_residual))]
year_max_relative_residual_val=max(np.abs(relative_cyclical_residual))
print(f"The year with the biggest fluctuation under 'Relative Cyclic Residuals' is: {year_max_relative_residual}")
print(f"The biggest fluctuation under 'Relative Cyclic Residuals' is: {year_max_relative_residual_val}%")

if year_max_percent_trend == year_max_relative_residual:
    print("Yes, the year with the biggest fluctuation is the same for both measures.")
else:
    print("No, the year with the biggest fluctuation is different for both measures.")

import matplotlib.pyplot as plt

plt.scatter(year, boxes, label='Actual Boxes')
plt.plot(year, trend_values, color='red', label='Trend Line')
plt.xlabel("Year")
plt.ylabel("Boxes (in 10,000s)")
plt.title("Actual Boxes vs. Trend Line Over Time")
plt.legend()
plt.grid(True)
plt.show()

#2.———————————————————————————————————

“””The owner of The Pleasure-Glide Boat Company has compiled the following quarterly figures regarding the company's level of accounts receivable over the last 5 years (× $1,000):

	•	﻿﻿﻿﻿Calculate a 4-quarter centered moving average.
	•	﻿﻿﻿﻿Find the percentage of actual to moving average for each period.
	•	﻿﻿﻿﻿Determine the modified seasonal indices and the seasonal”””


years = [1991, 1992, 1993, 1994, 1995]
quarters = ['Spring', 'Summer', 'Fall', 'Winter']
data = [
    [102, 120, 90, 78],
    [110, 126, 95, 83],
    [111, 128, 97, 86],
    [115, 135, 103, 91],
    [122, 144, 110, 98]
]

import pandas as pd

flat_data = []
for i in range(len(years)):
    for j in range(len(quarters)):
        flat_data.append({'Year': years[i], 'Quarter': quarters[j], 'Sales (Units)': data[i][j]})

df = pd.DataFrame(flat_data)
print(df)

# 4-Quarter Moving Total
df['4-Quarter Moving Total'] = df['Sales (Units)'].rolling(window=4,center=True).sum()

# 4-Quarter Moving Average
df['4-Quarter Moving Avg'] = df['4-Quarter Moving Total'] / 4

# 4-Quarter Centered Moving Average
df['4-Quarter Centered Moving Avg'] = df['4-Quarter Moving Avg'].rolling(window=2).mean().shift(-1)

# % of actual to Moving Average Value
df['% of actual to MAV'] = df['Sales (Units)'] / df['4-Quarter Centered Moving Avg']*100

df = df[['Year', 'Quarter', 'Sales (Units)', '4-Quarter Moving Total', '4-Quarter Moving Avg', '4-Quarter Centered Moving Avg', '% of actual to MAV']]

print("\nResults Table:")
display(df)

df1 = df.pivot_table(index='Year', columns='Quarter', values='% of actual to MAV')

quarter_order = ['Spring', 'Summer', 'Fall', 'Winter']
df1 = df1.reindex(columns=quarter_order)
display(df1)

import numpy as np

for col in df1.columns:
    values = df1[col].dropna()

    if len(values) >= 3:
        min_val = np.min(values)
        max_val = np.max(values)

        remaining_values = values[(values != min_val) & (values != max_val)]
        mean_of_remaining = np.mean(remaining_values)

        print(f"Quarter: {col}")
        print(f"Min (excluding NaN): {min_val:.2f}")
        print(f"Max (excluding NaN): {max_val:.2f}")
        print(f"Mean of remaining values (excluding min, max, and NaN): {mean_of_remaining:.2f}\n")
    elif len(values) > 0:
        print(f"Quarter: {col}")
        print(f"Not enough non-NaN values (less than 3) to exclude min and max. Values: {values.tolist()}\n")
    else:
        print(f"Quarter: {col}")
        print("No non-NaN values in this quarter.\n")

import numpy as np
import pandas as pd

df1 = df.pivot_table(index='Year', columns='Quarter', values='% of actual to MAV')

quarter_order = ['Spring', 'Summer', 'Fall', 'Winter']
df1 = df1.reindex(columns=quarter_order)

mean_values = {}

for col in df1.columns:
    values = df1[col].dropna()

    if len(values) >= 3:
        min_val = np.min(values)
        max_val = np.max(values)

        remaining_values = values[(values != min_val) & (values != max_val)]
        mean_of_remaining = np.mean(remaining_values)

        mean_values[col] = mean_of_remaining

    elif len(values) > 0:
        print(f"Quarter: {col}")
        print(f"Not enough non-NaN values (less than 3) to exclude min and max. Values: {values.tolist()}\n")
    else:
        print(f"Quarter: {col}")
        print("No non-NaN values in this quarter.\n")

print("Mean values (excluding min, max, and NaN) for each quarter:")
for quarter, mean_val in mean_values.items():
    print(f"{quarter}: {mean_val:.2f}")

sum_of_indices=sum(mean_values.values())
print("\nSum of Indices:")
print(sum_of_indices)

adjusting_const=400/sum_of_indices
print("\nAdjusting Constant:")
print(adjusting_const)

seasonal_indices = mean_values.copy()

for quarter in seasonal_indices:
    seasonal_indices[quarter] *= adjusting_const

print("Adjusted seasonal indices:")
for quarter, seasonal_index in seasonal_indices.items():
    print(f"{quarter}: {seasonal_index:.2f}")


sum_seasonal_index=sum(seasonal_indices.values())
print("\nSum of Seasonal Indices:")
print(sum_seasonal_index)

mean_seasonal_index=sum_seasonal_index/4
print("\nMean Seasonal Index:")
print(mean_seasonal_index)

#3.------------------------------------------------------------------------------

years = [1992, 1993, 1994, 1995]
quarters = ['Spring', 'Summer', 'Fall', 'Winter']
data = [
    [5.6, 6.8, 6.3, 5.2],
    [5.7, 6.7, 6.4, 5.4],
    [5.3, 6.6, 6.1, 5.1],
    [5.4, 6.9, 6.2, 5.3],
]

import pandas as pd

flat_data = []
for i in range(len(years)):
    for j in range(len(quarters)):
        flat_data.append({'Year': years[i], 'Quarter': quarters[j], 'Sales (Units)': data[i][j]})

df = pd.DataFrame(flat_data)

# 4-Quarter Moving Total
df['4-Quarter Moving Total'] = df['Sales (Units)'].rolling(window=4,center=True).sum()

# 4-Quarter Moving Average
df['4-Quarter Moving Avg'] = df['4-Quarter Moving Total'] / 4

# 4-Quarter Centered Moving Average
df['4-Quarter Centered Moving Avg'] = df['4-Quarter Moving Avg'].rolling(window=2).mean().shift(-1)

# % of actual to Moving Average Value
df['% of actual to MAV'] = df['Sales (Units)'] / df['4-Quarter Centered Moving Avg']*100

df = df[['Year', 'Quarter', 'Sales (Units)', '4-Quarter Moving Total', '4-Quarter Moving Avg', '4-Quarter Centered Moving Avg', '% of actual to MAV']]

print("\nResults Table:")
display(df)

#4.------------------------------------------------------------------------------

years = [1992, 1993, 1994, 1995]
quarters = ['Winter', 'Spring', 'Summer', 'Fall']
data = [
    [293, 246, 231, 282],
    [301, 252, 227, 291],
    [304, 259, 239, 296],
    [306, 265, 240, 300],
]

import pandas as pd

flat_data = []
for i in range(len(years)):
    for j in range(len(quarters)):
        flat_data.append({'Year': years[i], 'Quarter': quarters[j], 'Sales (Units)': data[i][j]})

df = pd.DataFrame(flat_data)

# 4-Quarter Moving Total
df['4-Quarter Moving Total'] = df['Sales (Units)'].rolling(window=4,center=True).sum()

# 4-Quarter Moving Average
df['4-Quarter Moving Avg'] = df['4-Quarter Moving Total'] / 4

# 4-Quarter Centered Moving Average
df['4-Quarter Centered Moving Avg'] = df['4-Quarter Moving Avg'].rolling(window=2).mean().shift(-1)

# % of actual to Moving Average Value
df['% of actual to MAV'] = df['Sales (Units)'] / df['4-Quarter Centered Moving Avg']*100

df = df[['Year', 'Quarter', 'Sales (Units)', '4-Quarter Moving Total', '4-Quarter Moving Avg', '4-Quarter Centered Moving Avg', '% of actual to MAV']]

print("\nResults Table:")
display(df)

df1 = df.pivot_table(index='Year', columns='Quarter', values='% of actual to MAV')
display(df1)
quarter_order = ['Winter', 'Spring', 'Summer', 'Fall']
df1 = df1.reindex(columns=quarter_order)
display(df1)

import numpy as np
import pandas as pd

mean_values = {}

for col in df1.columns:
    values = df1[col].dropna()

    if len(values) >= 3:
        min_val = np.min(values)
        max_val = np.max(values)

        remaining_values = values[(values != min_val) & (values != max_val)]
        mean_of_remaining = np.mean(remaining_values)

        mean_values[col] = mean_of_remaining

    elif len(values) > 0:
        print(f"Quarter: {col}")
        print(f"Not enough non-NaN values (less than 3) to exclude min and max. Values: {values.tolist()}\n")
    else:
        print(f"Quarter: {col}")
        print("No non-NaN values in this quarter.\n")

print("Mean values (excluding min, max, and NaN) for each quarter:")
for quarter, mean_val in mean_values.items():
    print(f"{quarter}: {mean_val:.2f}")

sum_of_indices=sum(mean_values.values())
print("\nSum of Indices:")
print(sum_of_indices)

adjusting_const=400/sum_of_indices
print("\nAdjusting Constant:")
print(adjusting_const)

seasonal_indices = mean_values.copy()

for quarter in seasonal_indices:
    seasonal_indices[quarter] *= adjusting_const

print("Adjusted seasonal indices:")
for quarter, seasonal_index in seasonal_indices.items():
    print(f"{quarter}: {seasonal_index:.2f}")


sum_seasonal_index=sum(seasonal_indices.values())
print("\nSum of Seasonal Indices:")
print(sum_seasonal_index)

mean_seasonal_index=sum_seasonal_index/4
print("\nMean Seasonal Index:")
print(mean_seasonal_index)

seasonal_index=[]
for i in range(4):
    seasonal_index.append(seasonal_indices[quarter_order[i]]/100)
print(seasonal_index)

data_d = [[0 for _ in range(len(data[0]))] for _ in range(len(data))]

for i in range(len(data)):
  for j in range(len(data[i])):
    data_d[i][j]=data[i][j]/seasonal_index[j]

print("\nDeseasonalized data")
print(data_d)

original_data_flat = [item for sublist in data for item in sublist] # flattening to a single list

deseasonalized_data_flat = [item for sublist in data_d for item in sublist]

years_flat = [y for y in years for q in quarters]
quarters_flat = quarters * len(years)

quarter_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
quarter_numeric = [quarter_map[q] for q in quarters_flat] # mapping each quarter to a numeric value in(0,1,2,3)

# Combining year and quarter for x-axis
x_values = [y + (q_num / 4) for y, q_num in zip(years_flat, quarter_numeric)]
x_labels = [f'{y} {q}' for y, q in zip(years_flat, quarters_flat)]

time_points = np.arange(len(deseasonalized_data_flat))
print(time_points)
slope_deseasonalized, intercept_deseasonalized = np.polyfit(time_points, deseasonalized_data_flat, 1)
trend_deseasonalized = slope_deseasonalized * time_points + intercept_deseasonalized

plt.figure(figsize=(15, 6))

#original data
plt.plot(x_values, original_data_flat, marker='o',color='black', linestyle='-', label='Original Data')

#deseasonalized data
plt.plot(x_values, deseasonalized_data_flat, color='Purple', marker='o', linestyle='-', label='Deseasonalized Data')

#trend line
plt.plot(x_values, trend_deseasonalized, color='cyan', label='Trend Line')
plt.xlabel("Year and Quarter")
plt.ylabel("Sales (Units)")
plt.title("Original and Deseasonalized Sales Data Over Time")
plt.xticks(x_values, x_labels, rotation=45, ha='right')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#5.------------------------------------------------------------------------------

import pandas as pd
import numpy as np

df = pd.read_csv('500040.csv') # Aditya_birla Real Estate Industries

dates = np.random.randint(1, len(df), 150)

selected_df = df.iloc[dates].copy()

selected_df.loc[:, 'Rate of Return'] = ((selected_df['Close Price'] - selected_df['Open Price']) / selected_df['Open Price']) * 100

f_df = selected_df[['Open Price', 'Close Price', 'Rate of Return']].copy()
display(f_df.head())

# linear regression

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

x_cubic = np.column_stack((np.ones(len(x)), x, x*2, x*3))

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

y_pred_cubic = intercept_cubic + slope_cubic_x * x + slope_cubic_x2 * (x*2) + slope_cubic_x3 * (x*3)

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

#6) -----------------------------------------------------------------------------

def get_season(month):
    if month in [3, 4, 5]:
        return '1'
    elif month in [6, 7, 8]:
        return '2'
    elif month in [9, 10, 11]:
        return '3'
    else:
        return '4'

monthly_data = pd.read_csv('/content/500040(1).csv')
monthly_data['Date'] = pd.to_datetime(monthly_data['Month'], format='%B-%Y')

monthly_data['Rate of Return'] = ((monthly_data['Close Price'] - monthly_data['Open Price']) / monthly_data['Open Price'])*100

df_stock_monthly_data = monthly_data[(monthly_data['Date'].dt.year >= 2020) & (monthly_data['Date'].dt.year <= 2024)].copy()

display(df_stock_monthly_data)

df_stock_monthly_data['Year'] = df_stock_monthly_data['Date'].dt.year

df_stock_monthly_data['Quarter Num'] = df_stock_monthly_data['Date'].dt.month.apply(get_season)

df_stock_monthly_data['Quarter_Num'] = df_stock_monthly_data['Quarter Num'].astype(int)

df_seasonal_returns_original = df_stock_monthly_data.groupby(['Year', 'Quarter Num'])['Rate of Return'].sum().unstack()

display(df_seasonal_returns_original)
df_seasonal_returns = []

for year, row in df_seasonal_returns_original.iterrows():
    for quarter, value in row.items():
        df_seasonal_returns.append({'Year': year, 'Quarter_Num': quarter, 'Rate of Return': value})

df_seasonal_returns = pd.DataFrame(df_seasonal_returns)

display(df_seasonal_returns)

df_seasonal_returns['4_Quarter_MA'] = df_seasonal_returns['Rate of Return'].rolling(window=4, center=True).mean()

quarter_centered_MA = []

for i in range(len(df_seasonal_returns['4_Quarter_MA'])):
    quarter_centered_MA.append(sum(df_seasonal_returns['4_Quarter_MA'][i:i+2])/2)

df_seasonal_returns['4_Quarter_Centered_MA'] = quarter_centered_MA

percentage_actual_MA = []

for i in range(len(df_seasonal_returns['Rate of Return'])):
    percentage_actual_MA.append((df_seasonal_returns['Rate of Return'][i] / df_seasonal_returns['4_Quarter_Centered_MA'][i]) * 100)

df_seasonal_returns['Percentage_Actual_MA'] = percentage_actual_MA

modified_seasonal_indices = df_seasonal_returns.groupby('Quarter_Num')['Percentage_Actual_MA'].apply(lambda x: x.drop([x.idxmin(), x.idxmax()]).median())

print("Modified Seasonal Indices (after removing min/max):")
print(modified_seasonal_indices)

sum_modified_seasonal_indices = modified_seasonal_indices.sum()

print("\nSum of Modified Seasonal Indices:", sum_modified_seasonal_indices)

adjustment_factor = 400 / sum_modified_seasonal_indices

print("\nAdjustment factor:", adjustment_factor)

seasonal_indices = modified_seasonal_indices * adjustment_factor

print("\nSeasonal Indices:")
print(seasonal_indices)

df_seasonal_returns['Seasonal_Index'] = df_seasonal_returns['Quarter_Num'].map(seasonal_indices)

df_seasonal_returns['Seasonal_Index / 100'] = df_seasonal_returns['Seasonal_Index'] / 100

df_seasonal_returns['Deseasonalized_Data'] = df_seasonal_returns['Rate of Return'] / df_seasonal_returns['Seasonal_Index / 100']

df_seasonal_returns["Year_Quarter"] = df_seasonal_returns["Year"].astype(str) + "Q" + df_seasonal_returns["Quarter_Num"].astype(str)

display(df_seasonal_returns)

plt.figure(figsize=(12,6))
plt.plot(df_seasonal_returns["Year_Quarter"], df_seasonal_returns['Rate of Return'], label = "Actual")
plt.plot(df_seasonal_returns["Year_Quarter"], df_seasonal_returns['Deseasonalized_Data'], label = "Deseasonalized_Data")
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.show()