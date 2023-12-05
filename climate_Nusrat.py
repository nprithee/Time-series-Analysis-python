

#Import all libraries
#%%
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from keras.models import Sequential
from keras.layers import LSTM, Dense

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#LOAD CSV
#%%
climate_df = pd.read_csv(r'/Users/nusratprithee/Documents/Time_Series_Project/DailyDelhiClimateTrain.csv')

# Set Date as index
climate_df.set_index('date',inplace=True)

climate_df.head()


climate_df.info()

climate_df.isnull().sum()

climate_df.describe().T


## Data Cleaning and Visualization

# Data Visualization
# In this section, I am going to do various visualization techniques to learn more about our dataset.

#%%
plt.figure(figsize=(15, 5))
climate_df['meantemp'].plot()
plt.title("Visualize Date and Meantemp")


# Observation: From the above plot, we observe that temperature rises exponentially in the first quarter of the year, remains high during the second quarter, seems to be constant during third quarted, and starts to reduce during the last quarter. This is the case in all yearly between 2013 and 2027.


#%%

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))
sns.despine()

climate_df['meantemp'].plot(ax=axs[0])
axs[0].set_title(f'Feature: meantemp', fontsize=14)
axs[0].set_ylabel(ylabel='meantemp', fontsize=12)

climate_df['humidity'].plot(ax=axs[1])
axs[1].set_title(f'Feature: humidity', fontsize=14)
axs[1].set_ylabel(ylabel='humidity', fontsize=12)

climate_df['wind_speed'].plot(ax=axs[2])
axs[2].set_title(f'Feature: wind_speed', fontsize=14)
axs[2].set_ylabel(ylabel='wind_speed', fontsize=12)

climate_df['meanpressure'].plot(ax=axs[3])
axs[3].set_title(f'Feature: meanpressure', fontsize=14)
axs[3].set_ylabel(ylabel='meanpressure', fontsize=12)

plt.tight_layout()
plt.show()


# Observation: The tiemseries has constant variation and is stationary. Notably, meanpreassure seems to have some abnormal behaviour between 2016 and 2017. This abnormality is due to the fact there are outliers in our the time series.

# Remove Outliers

#%%

def remove_outlier(df, col):
    """
    Parameters
    ==========
    df - Input dataframe
    col - col to remove outliers
    ==========
    Return
    ==========
    Dataframe with column whose outliers have been removed
    """
    values = df[col].values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (values > low) & (values < high)
    return df.loc[mask]


#%%

climate_df.columns

#%%

climate_df = remove_outlier(df=climate_df, col='meanpressure')

plt.figure(figsize=(10, 4))
climate_df['meanpressure'].plot(y='meanpressure')


# Observation:
# The problem seem to have been fixed.


#PACF of dependent variables
#%%


cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
data = climate_df[cols]

# PACF plot for each column
for col in data.columns:
    fig, ax = plt.subplots(figsize=(5, 2))
    sm.graphics.tsa.plot_pacf(data[col], ax=ax, lags=30, title=f'PACF for {col}')
    plt.show()


# From PACF of the depentend variables above, we can see that, corrlarion decays to zero and this shows that it is constant and stationery.

#Correlation matrix

#%%
# Heatmap
cliamate_corr = climate_df[cols].corr()
sns.heatmap(cliamate_corr, vmax=1, vmin=-1, annot=True)

print(f"Pearson Correlation Matrix\n{cliamate_corr}")


# From above heatmap, humidity, meanpressure, and windpeed have high correlation as compared to other features. We can also see that, wind_speed increaseses with decrease in humidity and meanpressure. Let's confirm this by performing decomposition below;

##Seasonal Decomposition

#%%


fig, ax =plt.subplots(nrows=4,ncols=4,figsize=(20,15))

for i, col in enumerate(cols):
    
    res = seasonal_decompose(climate_df[col], period=52, model='additive', extrapolate_trend='freq')
    
    ax[0,i].set_title(f'Decomposition of {col}', fontsize=14)
    res.observed.plot(ax=ax[0,i], legend=False)
    ax[0,i].set_ylabel('Observed', fontsize=12)

    res.trend.plot(ax=ax[1,i], legend=False)
    ax[1,i].set_ylabel('Trend', fontsize=12)

    res.seasonal.plot(ax=ax[2,i], legend=False)
    ax[2,i].set_ylabel('Seasonal', fontsize=12)
    
    res.resid.plot(ax=ax[3,i], legend=False)
    ax[3,i].set_ylabel('Residual', fontsize=12)
    
plt.tight_layout()

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

#%%
##Feature Selection Using RANDOM FOREST

#%%
# Drop any rows with missing values
climate_df.dropna(inplace=True)

# Split the data into features and target variable
X = climate_df.drop(['meantemp'], axis=1)  # Features
y = climate_df['meantemp']                 # Target variable

# Train a random forest regressor model
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X, y)

# Sort feature importances in descending order
importances = sorted(zip(X.columns, rf.feature_importances_), key=lambda x: x[1], reverse=True)

# Print feature importances in ranked order
for feature, importance in importances:
    print(f'{feature}: {importance:.3f}')


## Data Spliting
# training:testing = 80:20
# The aim of this project is to predict meantemp in new Delhi.

#%%
temp_df = pd.DataFrame(climate_df['meantemp'].copy(), columns=['meantemp'])
temp_df.head()
temp_df.value_counts()


#%%
train_size = int(len(temp_df) * 0.8)
training = temp_df.iloc[0:train_size]
testing = temp_df.iloc[train_size:len(temp_df)]

training[:5]
testing[:5]


#%%

# Prepare Dataset
def climate_dataset(X, y, window):
    XX = []
    YY = []
    for i in range(len(X) - window):
        vals = X.iloc[i: i+window].values
        XX.append(vals)
        YY.append(y.iloc[i+window])
    
    return np.array(XX), np.array(YY)


#%%

X = training['meantemp']
y = testing['meantemp']
X_train, y_train = climate_dataset(training, X, 5)
X_test, y_test = climate_dataset(testing, y, 5)


## MODELING
### Baseline Models (average, naive, drift, simple,exponential smoothing)

#%%
# Define the seasonal window length
window_length = 365

# Fit the model to the training set using Exponential Smoothing
model = ExponentialSmoothing(y_train, seasonal_periods=window_length, trend='add', seasonal='add')
fit = model.fit()

# Make predictions on the test set
preds = fit.forecast(len(X_test))

# Compute the MAE of the predictions
mse_exp_smoothing = (abs(y_test - preds)).mean()


#%%
# Compute the average value of the training target variable
y_avg = y_train.mean()
preds_avg = [y_avg] * len(y_test)
mse_avg = mean_squared_error(y_test, preds_avg)


#%%
# Use the last observed value as a naive baseline
last_observed = y_train[-1]
preds_naive = [last_observed] * len(y_test)
mse_naive = mean_squared_error(y_test, preds_naive)


#%%
# Use a linear drift model as a baseline
slope = (y_train[-1] - y_train[0]) / len(y_train)
intercept = y_train[0]
preds_drift = [(slope * i + intercept) for i in range(len(y_test))]
mse_drift = mean_squared_error(y_test, preds_drift)


#%%
# Use a simple model that forecasts the next value as the same as the current value
preds_simple = [y_train[-1]] * len(y_test)
mse_simple = mean_squared_error(y_test, preds_simple)


#%%
# Print the MAEs for each model
print(f"MAE for Average: {mse_avg:.4f}")
print(f"MAE for Naive: {mse_naive:.4f}")
print(f"MAE for Drift: {mse_drift:.4f}")
print(f"MAE for Simple: {mse_simple:.4f}")
print(f"MAE for ExponentialSmoothing: {mse_exp_smoothing:.4f}")

## Multi Linear Regression

#%%
# Reshape to 2d
X_train_2d = X_train.reshape((X_train.shape[0], X_train.shape[1]))

X_test_2d = X_test.reshape((X_test.shape[0], X_test.shape[1]))


#%%
# Fit a multiple linear regression model
model = LinearRegression().fit(X_train_2d, y_train)

# Get the predictions on the test set
y_pred = model.predict(X_test_2d)


#%%

accuracy_mtl = r2_score(y_test, y_pred) * 100


#%%


print(f"Multilinear Regression: {accuracy_mtl:.2f}  %")


#%%

# Get the last row of X_test_2d
x_pred_2d = X_test_2d[-1, :].reshape(1, -1)

# Predict the target value for the last row
y_pred_one_step = model.predict(x_pred_2d)

# Compare with the corresponding value in y_test
y_true_one_step = y_test[-1]
print(f"True value: {y_true_one_step}, predicted value: {y_pred_one_step}")


## Hypothesis Test

#%%


# Perform the F-test on the test set
f_val, p_val = f_regression(X_test_2d, y_test)

# Print the F-test results
print(f"F-value: {f_val}")
print(f"P-value: {p_val}")


#%%


# Add a constant term to the X_train matrix
X_train_const = sm.add_constant(X_train_2d)

# Fit a multiple linear regression model using statsmodels
model_sm = sm.OLS(y_train, X_train_const).fit()

# Perform the t-test on the test set
X_test_const = sm.add_constant(X_test_2d)
t_test = model_sm.t_test(X_test_const)

# Print the t-test results
print(t_test.summary())


#%%

# IC, BIC, RMSE, R-Square, Adjusted R-Squared

# Calculate the mean squared error and R-squared on the test set
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (X_test.shape[0] - 1) / (X_test.shape[0] - X_test.shape[1] - 1)

# Calculate AIC and BIC
n = X_test.shape[0]
k = X_test.shape[1]
aic = n * np.log(mse) + 2 * k
bic = n * np.log(mse) + k * np.log(n)

# Print the evaluation metrics
print(f"Multi Linear Regression Evaluation\n********************")
print(f"RMSE: {np.sqrt(mse)}")
print(f"R-Square: {r2}")
print(f"Adjusted R-Square: {adj_r2}")
print(f"AIC: {aic}")
print(f"BIC: {bic}")


#%%


# Calculate the residuals
residuals = y_test - y_pred

# Plot the ACF of the residuals
plot_acf(residuals)


#%%


# Q-Variance

# Perform the Breusch-Pagan test on the residuals
bp_test = het_breuschpagan(residuals, X_test_2d)

# Print the Q-variance test results
print(f"Lagrange multiplier statistic: {bp_test[0]}")
print(f"P-value: {bp_test[1]}")
print(f"F-statistic: {bp_test[2]}")
print(f"F p-value: {bp_test[3]}")


#%%


# Variance and mean of the residual
residual_mean = np.mean(residuals)
residual_variance = np.var(residuals)

print(f"Residual mean: {residual_mean}")
print(f"Residual variance: {residual_variance}")


## ARMA and SARIMA

# NotImplementedError: 
# statsmodels.tsa.arima_model.ARMA and statsmodels.tsa.arima_model.ARIMA have
# been removed in favor of statsmodels.tsa.arima.model.ARIMA (note the .
# between arima and model) and statsmodels.tsa.SARIMAX.
# 
# statsmodels.tsa.arima.model.ARIMA makes use of the statespace framework and
# is both well tested and maintained. It also offers alternative specialized
# parameter estimators.

#%%


# Fit SARIMA model
sarima_model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 11))
sarima_results = sarima_model.fit()

# Make predictions using the fitted model
y_pred_sarima = sarima_results.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)


#%%


# Fit an ARIMA model to the training data and make predictions on the test set
model = ARIMA(y_train, order=(1, 0, 0))
fit = model.fit()
preds_arima = fit.forecast(len(y_test))


# EVALUATE SARIMA AND ARIMA

#%%


sarima_mse = mean_squared_error(y_test, y_pred_sarima)
mse_arima = mean_squared_error(y_test, preds_arima)

print("Sarima MAE: ", sarima_mse)
print(f"MAE for ARIMA: {mse_arima:.4f}")


# Future predictions

#%%

# number of periods to forecast
n_periods = 30

# Make future predictions
y_pred_future = sarima_results.forecast(steps=n_periods)

print(y_pred_future)


#%%


# Plot the predicted future values against true values
plt.plot(y_test, label='True Values')
plt.plot(y_pred_future, label='Predicted Values')
plt.title("SARIMA Future Prediction")
plt.legend()
plt.show()


#%%


# Forecast the future values
forecast_arima = fit.forecast(n_periods)

# Print the predicted values
print(forecast_arima)

# Plot the predicted values against the original data
plt.plot(y)
plt.plot(np.arange(len(y), len(y)+n_periods), forecast_arima, color='red')
plt.title("ARIMA Future Prediction")
plt.show()


# LSTM Model

#%%


def LSTMModel(x_train):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


#%%


# Instantiate the LSTM model
lstm_model = LSTMModel(X_train)


#%%


lstm_model.summary()


#%%



# Fit the model to the training data
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)


#%%


# Predict the values using the LSTM model
y_pred_lstm = lstm_model.predict(X_test)

# Compute the Mean Squared Error (MSE)
mse_lstm = mean_squared_error(y_test, y_pred_lstm)
print(f"LSTM MSE: {mse_lstm}")


#%%


# Visualize the predicted values against the actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred_lstm, label='LSTM Predicted')
plt.legend()
plt.show()


#%%


# Prepare the input data for forecasting
n_steps = 5
n_features = 1
last_observation = X_test[-1]
forecast_input = last_observation.reshape((1, n_steps, n_features))

# Forecast the next 30 days of temperatures
n_forecast_days = 30
forecast = []
for i in range(n_forecast_days):
    y_pred = lstm_model.predict(forecast_input)[0]
    forecast.append(y_pred)
    # Update the forecast input with the latest prediction
    forecast_input = np.append(forecast_input[:,1:,:], [[y_pred]], axis=1)

# Plot the forecasted values
plt.plot(range(len(y)), y, label='True values')
plt.plot(range(len(y)-n_forecast_days, len(y)), forecast, label='Forecasted values')
plt.legend()
plt.show()


# Forecast Function

#%%


def sarima_forecast(model, n_periods):
    """
    Function to make future predictions using a fitted SARIMA model.

    Args:
    model: Fitted SARIMA model.
    n_periods: Number of periods to forecast into the future.

    Returns:
    Array of future predictions.
    """

    # Make future predictions
    y_pred_future = model.forecast(steps=n_periods)

    return y_pred_future


#%%


# Apply Sarima Model
# Specify the number of periods to forecast
n_periods = 24

# Use the SARIMA model to forecast future values
forecast = sarima_forecast(model=sarima_results, n_periods=n_periods)

# Print the forecasted values
print(forecast)


# %%
