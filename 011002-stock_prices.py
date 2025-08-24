import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
import io
import yfinance as yf

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import linear_model

from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import seaborn as sns
from tensorflow.python.ops.gen_array_ops import tile_grad
from my_statsmodels import *

sns.set_theme()


# load the data
df0 = pd.read_csv("meta.csv")
print(df0.head())

# treat the data, formating, cleanup
print(df0.head())
# Drops or erases the first two rows or records
df0_cleaned = df0.drop([0, 1])
df0_cleaned = ((df0_cleaned.rename(columns={"Price": "Date"})).reset_index())
# The old index is dropped
df_meta = df0_cleaned.drop(['index'],axis=1)
df_meta = df0_cleaned.drop(['Unnamed: 0'],axis=1)
# The "Close", "Date" column are converted from string to numeric and datetime respectively
df_meta['Close'] = pd.to_numeric(df_meta['Close'], errors="coerce")
df_meta['Date'] = pd.to_datetime(df_meta['Date'])
print(df_meta.head())

# order data types
price = df_meta['Close'].values.reshape(-1,1)
date = df_meta['Date'].values.reshape(-1,1)
index = df_meta.index.values.reshape(-1,1)
log_price = np.log(price)
fig, axs = plt.subplots(1, 2)
#axs0.title("META/FB stock price over time")
axs[0].plot(date, price)
#axs1.title("log of META/FB stock price over time")
axs[1].plot(date, log_price)
plt.show()

coefficients,intercept, y1_hat = q_regresion(index, log_price, degree=1)
print("Linear fit: ",coefficients,intercept) # To print the coefficient estimate of the series.
coefficients,intercept, y2_hat = q_regresion(index, log_price, degree=2)
print("Cuadratic fit: ",coefficients,intercept) # To print the coefficient estimate of the series.
coefficients,intercept, y3_hat = q_regresion(index, log_price, degree=3)
print("Cubic fit: ",coefficients,intercept) # To print the coefficient estimate of the series.

plt.plot(date, log_price, label='log stock price (original data)')
plt.plot(date, y1_hat, 'r', label='Linear fitted line')
plt.plot(date, y2_hat, 'r', label='Quadratic fitted line')
plt.plot(date, y3_hat, 'r', label='Cubic fitted line')
plt.legend()
plt.show()
linear_residuals = log_price - y1_hat
quadratic_residuals = log_price - y2_hat
cubic_residuals = log_price - y3_hat
plt.plot(date, linear_residuals, '.')
plt.plot(date, quadratic_residuals, '.')
plt.plot(date, cubic_residuals, '.')
plt.show()
print("MSE with linear fit:", np.mean((linear_residuals)**2))
print("AIC:", evaluate_AIC(1, linear_residuals))
print("BIC:", evaluate_BIC(1, linear_residuals))
print("MSE with quadratic fit:", np.mean((quadratic_residuals)**2))
print("AIC:", evaluate_AIC(1, quadratic_residuals))
print("BIC:", evaluate_BIC(1, quadratic_residuals))
print("MSE with cubic fit:", np.mean((linear_residuals)**2))
print("AIC:", evaluate_AIC(1, cubic_residuals))
print("BIC:", evaluate_BIC(1, cubic_residuals))

acf1_values_full, confint, qstat, pvalues = acf(linear_residuals, nlags=30,alpha=0.05, qstat=True)
print(acf1_values_full)
pacf1_values_full, confint = pacf(linear_residuals, nlags=30, alpha=0.05)
print(pacf1_values_full)
acf2_values_full, confint, qstat, pvalues = acf(quadratic_residuals, nlags=30, alpha=0.05, qstat=True)
print(acf2_values_full)
pacf2_values_full, confint = pacf(quadratic_residuals, nlags=30, alpha=0.05)
print(pacf2_values_full)
acf3_values_full, confint, qstat, pvalues = acf(cubic_residuals, nlags=30, alpha=0.05, qstat=True)
print(acf3_values_full)
pacf3_values_full, confint = pacf(cubic_residuals, nlags=30, alpha=0.05)
print(pacf3_values_full)

fig, axs = plt.subplots(3, 2)
sm.graphics.tsa.plot_acf(linear_residuals, lags=30, ax=axs[0,0], title="ACF linear")
sm.graphics.tsa.plot_pacf(linear_residuals, lags=30, ax=axs[0,1], title="PACF linear")
sm.graphics.tsa.plot_acf(quadratic_residuals, lags=30, ax=axs[1,0], title="ACF Quadratic")
sm.graphics.tsa.plot_pacf(quadratic_residuals, lags=30, ax=axs[1,1], title="PACF Quadratic")
sm.graphics.tsa.plot_acf(cubic_residuals, lags=30, ax=axs[2,0], title="ACF Cubic")
sm.graphics.tsa.plot_pacf(cubic_residuals, lags=30, ax=axs[2,1], title="PACF Cubic")
plt.show()

min_aic_index1, min_bic_index1, _, _ = grid_search_ARIMA(linear_residuals, range(4), range(4), verbose=True)
min_aic_index2, min_bic_index2, _, _ = grid_search_ARIMA(quadratic_residuals, range(4), range(4), verbose=True)
min_aic_index3, min_bic_index3, _, _ = grid_search_ARIMA(cubic_residuals, range(4), range(4), verbose=True)

quick_summary_ARIMA(linear_residuals,min_aic_index1,min_bic_index1)
quick_summary_ARIMA(quadratic_residuals,min_aic_index2,min_bic_index2)

train_test_split = int(len(price) * 0.8)
train_price, test_price = quadratic_residuals[:train_test_split], quadratic_residuals[train_test_split:]
train_date, test_date = date[:train_test_split], date[train_test_split:]
assert(len(train_date) + len(test_date) == len(date))

## First, let's see how this does with the AIC selected values.
arma = ARIMA(train_price, order=min_aic_index2).fit()
print(arma.summary())
fcast = arma.get_forecast(len(test_price)).summary_frame()
forecast_means = fcast['mean'].values.reshape(-1,1)
arma_predictions = arma.predict()
predicted_values = arma_predictions.reshape(-1,1)
test_set_mse = np.mean((forecast_means.reshape(test_price.shape) - test_price)**2)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(date, quadratic_residuals, label='stock price, converted to stationary time series')
ax.plot(train_date, predicted_values, 'r', label='fitted line')
ax.plot(test_date, forecast_means, 'k--', label='mean forecast')
ax.fill_between(test_date.flatten(), fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.legend();
print("Test set mean squared error: ", test_set_mse)

## Now, let's see how this does with the BIC selected values.
arma = ARIMA(train_price, order=min_bic_index2).fit()
print(arma.summary())
fig, ax = plt.subplots(figsize=(15, 5))

# Construct the forecasts
fcast = arma.get_forecast(len(test_price)).summary_frame()

arma_predictions = arma.predict()
ax.plot(date, quadratic_residuals, label='stock price, converted to stationary time series')
predicted_values = arma_predictions.reshape(-1,1)
ax.plot(train_date, predicted_values, 'r', label='fitted line')
forecast_means = fcast['mean'].values.reshape(-1,1)
test_set_mse = np.mean((forecast_means.reshape(test_price.shape) - test_price)**2)
ax.plot(test_date, forecast_means, 'k--', label='mean forecast')
ax.fill_between(test_date.flatten(), fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.legend();
print("Test set mean squared error: ", test_set_mse)

#collapsed = (df_meta['Date'].Grouper(freq ='M')).mean()
#collapsed = df_meta.groupby(pd.Grouper(key='Date', freq='M'))
df1 = df_meta.resample('M',on='Date')['Close'].mean()
#df2 = df_meta.resample('M',on='Date')['High'].mean()
#print(df2.head())
collapsed = df1.reset_index()
month_date = collapsed['Date'].dt.date.values.reshape(-1,1)
month_price = collapsed['Close'].values.reshape(-1,1)
month_log_price = np.log(month_price)
plt.plot(month_date, month_log_price)
plt.title("log of Facebook stock price over time")
plt.show()

clf = linear_model.LinearRegression()
index = collapsed.reset_index().index.values.reshape(-1,1)
new_x = np.hstack((index, index **2))
clf.fit(new_x, month_log_price)
print(clf.coef_) # To print the coefficient estimate of the series.
month_quad_prediction = clf.predict(new_x)
plt.plot(month_date, month_log_price, label='original data')
plt.plot(month_date, month_quad_prediction, 'r', label='fitted line')
plt.legend()
plt.show()
month_quad_residuals = month_log_price - month_quad_prediction
plt.plot(month_date, month_quad_residuals, 'o')
plt.show();
print("MSE with quadratic fit:", np.mean((month_quad_residuals)**2))

sm.graphics.tsa.plot_acf(month_quad_residuals, lags=14)
plt.show()
sm.graphics.tsa.plot_pacf(month_quad_residuals, lags=14)
plt.show()

month_train_test = int(0.8 * len(month_date))
month_train, month_test = month_quad_residuals[:month_train_test], month_quad_residuals[month_train_test:]
month_date_train, month_date_test = month_date[:month_train_test], month_date[month_train_test:]

min_aic_index, min_bic_index, *other = grid_search_ARIMA(month_quad_residuals, range(4), range(4), verbose=True)
if min_aic_index == min_bic_index:
  arma = ARIMA(month_quad_residuals, order=min_bic_index).fit()
  print(arma.summary())
  arma_predictions = arma.predict()
  arma_residuals = month_quad_residuals.reshape(arma_predictions.shape) - arma_predictions
  arma_residuals = arma_residuals # Fitting AR 1 model means removing one observation
  plt.plot(month_quad_residuals, label='Residuals from fitted quadratic line')
  plt.plot(arma_predictions, 'r', label='fitted ARMA process')
  plt.legend()
  plt.show()
  plt.plot(arma_residuals, 'o')
  plt.show()
  print("Automatic selection finds model with AR {0}, MA {2}".format(*min_aic_index))
  print("MSE with selected model:", np.mean(arma_residuals**2))
else:
  print("AIC, BIC do not agree.")

arma = ARIMA(month_train, order=min_bic_index).fit()
fig, ax = plt.subplots(figsize=(15, 5))

# Construct the forecasts
fcast = arma.get_forecast(len(month_test)).summary_frame()

arma_predictions = arma.predict()
ax.plot(month_date, month_quad_residuals, label='original data')
predicted_values = arma_predictions.reshape(-1,1)
ax.plot(month_date_train, predicted_values, 'r', label='fitted line')
forecast_means = fcast['mean'].values.reshape(-1,1)
ax.plot(month_date_test, forecast_means, 'k--', label='mean forecast')
ax.fill_between(month_date_test.flatten(), fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
plt.legend();

test_set_mse = np.mean((forecast_means.reshape(month_test.shape) - month_test)**2)
print("Test set mean squared error: ", test_set_mse)