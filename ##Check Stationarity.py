##Check Stationarity

#ADF Test of stationary
#%%
result = sm.tsa.stattools.adfuller(climate_df['meantemp'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
 print('\t{}: {}'.format(key, value))

#Rolling mean and Varience
#%%
rolling_mean = climate_df['meantemp'].rolling(window=30).mean()
rolling_var = climate_df['meantemp'].rolling(window=30).var()

plt.figure(figsize=(15,5))
plt.plot(climate_df['meantemp'], color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_var, color='green', label='Rolling Variance')
plt.legend(loc='best')
plt.title('Rolling Mean and Variance')
plt.show()

#Transformation of the Data
#%%
climate_df['diff_1'] = climate_df['meantemp'].diff()
#Plot rolling mean and varience after transform
#%%
plt.figure(figsize=(15,5))
plt.plot(climate_df['diff_1'], color='blue',label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_var, color='green', label='Rolling Variance')
plt.legend(loc='best')
plt.title('Rolling Mean & Variance')
plt.show()  