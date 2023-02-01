#%%
import matplotlib.pyplot as plt 
import pandas as pd 
from statsmodels.tsa.stattools import adfuller
import datetime
# %%
#Here we get the data from github:
#url = "https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/blob/afebca0c811a3e63a06b95a850931610167cfa1e/tute1.csv"

#Here, we read the csv file and put int into a dataframe called df
df = pd.read_csv("tute1.csv")

#Here, we show the first 4 rows in the dataframe df
df.head(4)
# %%
#Question1
df.columns = ['Date', 'Sales', 'AdBudget', 'GDP']

df['Date'] = pd.to_datetime(df.Date, format = "%m/%d/%Y")
df = df.set_index("Date")

fig, ax = plt.subplots(figsize = (12,8))

ax.plot(df.Sales, label = "Sales")
ax.plot(df.AdBudget, label = "AdBudget")
ax.plot(df.GDP, label = "GDP")
plt.legend(labels = ["Sales", "AdBudget", "GDP"] )
ax.set_ylabel("USD($)")
ax.set_xlabel("Date")
ax.set_title("Sales, Ad Budget and GDP")
plt.grid()
plt.savefig("graph.png")
plt.show()
# %%
#Question 2
from statistics import mean, median, mode, stdev, variance
#Statistics Sale
sale_mean=df['Sales'].mean()
sale_var=df['Sales'].var()
sale_std=df['Sales'].std()
sale_median=df['Sales'].median()
print("The Sales mean is : %3d, and the variance is : %2d,with standard deviation :%2d,median :%2d" % (sale_mean,sale_var, sale_std,sale_median))

# Statistics AdBudget

AdBudget_mean=df['AdBudget'].mean()
AdBudget_var=df['AdBudget'].var()
AdBudget_std=df['AdBudget'].std()
AdBudget_median=df['AdBudget'].median()
print("The AdBudget mean is : %3d, and the variance is : %2d,with standard deviation :%2d,median :%2d" % (AdBudget_mean,AdBudget_var, AdBudget_std,AdBudget_median))

# Statistics GDP

GDP_mean=df['GDP'].mean()
GDP_var=df['GDP'].var()
GDP_std=df['GDP'].std()
GDP_median=df['GDP'].median()
print("The GDP mean is : %3d, and the variance is : %2d,with standard deviation :%2d,median :%2d" % (GDP_mean,GDP_var, GDP_std,GDP_median))


# %%
#Question3
def cal_rolling_mean_var(data, window):
   rolling_mean = data.rolling(window=window).mean()
   rolling_var = data.rolling(window=window).var()
   return rolling_mean, rolling_var
def plot_rolling_mean_var(data, window):
   rolling_mean, rolling_var = cal_rolling_mean_var(data, window)
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
   ax1.plot(rolling_mean)
   ax1.set_title('Rolling Mean')
   ax2.plot(rolling_var)
   ax2.set_title('Rolling Variance')
   plt.show()
   
# Plot the rolling mean and variance for Sales
plot_rolling_mean_var(df['Sales'], window=10)
# Plot the rolling mean and variance for AdBudget
plot_rolling_mean_var(df['AdBudget'], window=10)
# Plot the rolling mean and variance for GDP
plot_rolling_mean_var(df['GDP'], window=10)

# %%
#Question4

#Question5
from statsmodels.tsa.stattools import adfuller 
def ADF_Cal(x): 
    result = adfuller(x) 
    print("ADF Statistic: %f" %result[0]) 
    print('p-value: %f' % result[1]) 
    print('Critical Values:') 
    for key, value in result[4].items(): 
        print('\t%s: %.3f' % (key, value))
        
adfuller(df['Sales'])
adfuller(df['AdBudget'])
adfuller(df['GDP'])
        
#%%

#Question 6
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries): 
    print ('Results of KPSS Test:') 
    kpsstest = kpss(timeseries, regression='c', nlags="auto") 
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used']) 
    for key,value in kpsstest[3].items(): 
        kpss_output['Critical Value (%s)'%key] = value 
        print (kpss_output)
        
kpss(df['Sales'])
kpss(df['AdBudget'])
kpss(df['GDP'])

     
# %%
#Question 7
df= pd.read_csv("AirPassengers.csv")

print(df.head())
# %%
#Step1

df.columns = ['Month', 'Passengers']

df['Month'] = pd.to_datetime(df['Month'], format='%Y-%b-%d')
data=df.set_index(['Month'])
data.plot(figsize=(15, 6))
ax.set_ylabel("Numof Passenger")
ax.set_xlabel("Time")
ax.set_title("Month air passenger")
plt.grid()
plt.savefig("graph.png")
plt.show()


 # %%
 #Step2
Passengers_mean=df['Passengers'].mean()
Passengers_var=df['Passengers'].var()
Passengers_std=df['Passengers'].std()
Passengers_median=df['Passengers'].median()
print("The Passengers mean is : %3d, and the Passengers is : %2d,with standard deviation :%2d,median :%2d" % (Passengers_mean,Passengers_var, Passengers_std,Passengers_median))



# %%
#Step3

def cal_rolling_mean_var(data, window):
   rolling_mean = data.rolling(window=window).mean()
   rolling_var = data.rolling(window=window).var()
   return rolling_mean, rolling_var
def plot_rolling_mean_var(data, window):
   rolling_mean, rolling_var = cal_rolling_mean_var(data, window)
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
   ax1.plot(rolling_mean)
   ax1.set_title('Rolling Mean')
   ax2.plot(rolling_var)
   ax2.set_title('Rolling Variance')
   plt.show()
   
# Plot the rolling mean and variance for Passengers
plot_rolling_mean_var(df['Passengers'], window=10)
# %%
#Step4

#Step5
from statsmodels.tsa.stattools import adfuller 
def ADF_Cal(x): 
    result = adfuller(x) 
    print("ADF Statistic: %f" %result[0]) 
    print('p-value: %f' % result[1]) 
    print('Critical Values:') 
    for key, value in result[4].items(): 
        print('\t%s: %.3f' % (key, value))
        
adfuller(df['Passengers'])



# %%
#Step6
from statsmodels.tsa.stattools import kpss
def kpss_test(timeseries): 
    print ('Results of KPSS Test:') 
    kpsstest = kpss(timeseries, regression='c', nlags="auto") 
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used']) 
    for key,value in kpsstest[3].items(): 
        kpss_output['Critical Value (%s)'%key] = value 
        print (kpss_output)
        
kpss(df['Passengers'])

# %%
