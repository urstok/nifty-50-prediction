import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from time import time
import datetime
import warnings

symbol = "^NSEI"

symbol_name = "NIFTY_50_PREDICTION"

#import the data....tcs
tickersymbol = symbol
data = yf.Ticker(tickersymbol)
data

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=730)
tomorrow = end_date + datetime.timedelta(days=1)
ticker_symbol = symbol

# Fetch historical data
data = yf.Ticker(ticker_symbol)
prices = data.history(start=start_date, end=end_date)['Close']

print(prices)

#calculate returns
returns = prices.pct_change().dropna()
returns

#plot the stock prices
plt.figure(figsize=(10, 4))
plt.plot(prices)
plt.ylabel("Closing prices")
plt.xlabel("Date")
plt.title("Stock Closing Prices")
plt.show()

#plot the returns
plt.figure(figsize=(10, 4))
plt.plot(returns)
plt.ylabel("Closing prices")
plt.xlabel("Date")
plt.title("Returns")
plt.show()

import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming you have already fetched the returns or the time series data

# Plot ACF
sm.graphics.tsa.plot_acf(returns, lags=30)
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
sm.graphics.tsa.plot_pacf(returns, method='ols')
plt.xlabel('Lags')
plt.ylabel('Partial Autocorrelation')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# Assuming 'prices' contains your stock prices time series data

# Define the ARIMA model
order = (9, 0, 8)  # p=8, d=0, q=8
model = ARIMA(prices, order=order)

# Fit the model
fitted_model = model.fit()

# Print model summary
print(fitted_model.summary())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from datetime import datetime

# Generate some sample data
np.random.seed(0)
data = np.random.randn(100)
today_date = datetime.today().date()
index = pd.date_range(start=today_date, periods=100)

# Fit ARIMA model
model = ARIMA(data, order=(1,1,1))  # Example order, you should choose based on your data
fit_model = model.fit()

# Get predicted values
predicted_values = fit_model.predict()

# Calculate residuals
residuals = data - predicted_values

# Plot residuals
plt.figure(figsize=(10, 4))
plt.plot(index, residuals)
plt.title('Residuals Plot')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# Calculate MAE, MSE, RMSE
mae = mean_absolute_error(data, predicted_values)
mse = mean_squared_error(data, predicted_values)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

# Ljung-Box Test for autocorrelation of residuals
lb_stat, lb_p_value = acorr_ljungbox(residuals)
print("Ljung-Box Test p-value:", lb_p_value)

# Predicting 10 steps ahead
next_10_days_prices = fitted_model.forecast(steps=10)
print(next_10_days_prices)

data_series = next_10_days_prices
data_series = pd.Series(data_series)
data_series.index=[f"Day{i}" for i in range (1,len(data_series)+1)]
print(data_series)

#plot the stock prices
plt.figure(figsize=(20, 6))
plt.plot(data_series)
plt.ylabel("Closing prices")
plt.xlabel("Upcoming Day")
plt.title(symbol_name)
plt.show()
