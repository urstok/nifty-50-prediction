import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
import datetime

# Define the symbol and symbol name
symbol = "^NSEI"
symbol_name = "NIFTY_50_PREDICTION"

# Fetch historical data using yfinance
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=730)
data = yf.download(symbol, start=start_date, end=end_date)

# Extract closing prices
prices = data['Close']

# Define ARIMA model order
order = (9, 0, 8)  # Example order (p, d, q)

# Fit ARIMA model
model = ARIMA(prices, order=order)
fitted_model = model.fit()

# Forecasting next 10 days using fitted ARIMA model
forecast_steps = 10
forecasted_values = fitted_model.forecast(steps=forecast_steps)

# Generate date range for forecasted days
forecast_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

# Create DataFrame for forecasted values
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Prices': forecasted_values
})

# Add disclaimer message
disclaimer = """
-------------------------------------------------
Disclaimer: These are ARIMA model predictions and do not guarantee future results.
"""

# Save forecasted prices with disclaimer to CSV file
output_file = 'forecasted_prices_with_disclaimer.csv'
with open(output_file, 'w') as f:
    f.write(disclaimer.strip() + '\n\n')
    forecast_df.to_csv(f, index=False)

# Print forecasted values and disclaimer
print("Forecasted Prices for Next 10 Days:")
print(forecast_df)
print(disclaimer)

# Plot forecasted prices
plt.figure(figsize=(12, 6))
plt.plot(prices, label='Historical Prices')
plt.plot(forecast_dates, forecasted_values, label='Forecasted Prices', color='red', marker='o')
plt.title("Forecasted Stock Prices for Next 10 Days")
plt.xlabel("Date")
plt.ylabel("Closing Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
