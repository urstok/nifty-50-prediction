import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import datetime
from tabulate import tabulate
import warnings

# Suppress all warnings (not recommended unless you know the implications)
warnings.filterwarnings("ignore")

# Define the symbol and symbol name
symbol = "NIFTY_FIN_SERVICE.NS"
symbol_name = "FINNIFTY_PREDICTION"

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

# Add license and disclaimer message with specific headings in bold
license_and_disclaimer = """
-------------------------------------------------------------------------------------------------
### **License and Disclaimer**

#### **License**
The content and information provided in this document are intended solely
for educational and informational purposes.
This document is not for commercial use and is not to be shared publicly.
You may use this document for personal, non-commercial purposes only.

#### **Disclaimer**
The information contained herein does not constitute financial advice, investment advice,
trading advice, or any other sort of advice and should not be treated as such.
The content provided is for educational and informational purposes only.
Always seek the advice of a qualified financial advisor or
other professional regarding any financial decisions.

The author(s) of this document make no representations or warranties, express or implied,
as to the accuracy, completeness, or suitability of the information provided herein.
The author(s) will not be held liable for any errors or omissions,
or any losses, injuries, or damages arising from the use of this information.

Use this information at your own risk.
-------------------------------------------------------------------------------------------------
"""

# Print forecasted values and license/disclaimer
print("FINNIFTY Forecasted Prices for Next 10 Days:")
print(tabulate(forecast_df, headers='keys', tablefmt='grid', showindex=False))
print(license_and_disclaimer)
