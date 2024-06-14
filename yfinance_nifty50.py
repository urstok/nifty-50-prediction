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

# Function to get the next N trading days
def get_next_trading_days(start_date, n_days, holidays=None):
    if holidays is None:
        holidays = []
    trading_days = []
    current_date = start_date
    while len(trading_days) < n_days:
        current_date += datetime.timedelta(days=1)
        if current_date.weekday() < 5 and current_date not in holidays:  # Monday to Friday and not a holiday
            trading_days.append(current_date)
    return trading_days

# Forecasting next 10 trading days using fitted ARIMA model
forecast_steps = 10
forecasted_values = fitted_model.forecast(steps=forecast_steps)

# Generate date range for forecasted days
holidays = []  # Add official holidays here if needed
forecast_dates = get_next_trading_days(prices.index[-1].date(), forecast_steps, holidays)

# Create DataFrame for forecasted values
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Prices': forecasted_values
})

# Add license and disclaimer message
license_and_disclaimer = """
-------------------------------------------------
### License and Disclaimer

#### License
The content and information provided in this document are intended solely for educational and informational purposes. 
This document is not for commercial use and is not to be shared publicly. 
You may use this document for personal, non-commercial purposes only.

#### Disclaimer
The information contained herein does not constitute financial advice, investment advice, trading advice, 
or any other sort of advice and should not be treated as such. 
The content provided is for educational and informational purposes only. 
Always seek the advice of a qualified financial advisor or other professional regarding any financial decisions.

The author(s) of this document make no representations or warranties, express or implied, as to the accuracy, completeness, or suitability of the information provided herein. 
The author(s) will not be held liable for any errors or omissions, or any losses, injuries, or damages arising from the use of this information.

Use this information at your own risk.
-------------------------------------------------
"""

# Save forecasted prices with license and disclaimer to CSV file
output_file = 'forecasted_prices_with_disclaimer.csv'
with open(output_file, 'w') as f:
    # Write the license and disclaimer
    f.write(license_and_disclaimer.strip() + '\n\n')
    # Write the forecasted prices
    forecast_df.to_csv(f, index=False)

# Print forecasted values and license/disclaimer
print("Forecasted Prices for Next 10 Days:")
print(forecast_df)
print(license_and_disclaimer)

# Plot forecasted prices
plt.figure(figsize=(12, 6))
plt.plot(prices, label='Historical Prices')
plt.plot(forecast_dates, forecasted_values, label='Forecasted Prices', color='red', marker='o')
plt.title("Forecasted Stock Prices for Next 10 Trading Days")
plt.xlabel("Date")
plt.ylabel("Closing Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
