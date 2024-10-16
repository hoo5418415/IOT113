import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Function to convert ROC (Taiwanese calendar) to Gregorian calendar
def convert_roc_to_gregorian(roc_date):
    roc_year, month, day = map(int, roc_date.split('/'))
    gregorian_year = roc_year + 1911  # ROC year starts at 1911
    return datetime(gregorian_year, month, day)

# Load your CSV file
file_path = 'your_file_path.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Convert ROC dates to Gregorian calendar
data['Date'] = data['Date'].apply(convert_roc_to_gregorian)

# Set 'Date' as index for time series analysis
data.set_index('Date', inplace=True)

# Split data into training and test sets
train_data = data['Price'][:-30]  # Use all but the last 30 days for training
test_data = data['Price'][-30:]   # Last 30 days for testing

# Fit Auto Regressive model on training data
ar_model = AutoReg(train_data, lags=5).fit()

# Make predictions for the test set
predictions = ar_model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1, dynamic=False)

# Evaluate the model using RMSE
mse = mean_squared_error(test_data, predictions)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Forecast for the next 3 months (90 days)
forecast = ar_model.predict(start=len(data), end=len(data) + 90 - 1, dynamic=False)

# Plot actual prices and forecasted prices
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Price'], label='Actual Prices')

# Generate future dates for forecast
forecast_dates = pd.date_range(start=data.index[-1], periods=90, freq='B')  # 90 business days

# Plot forecasted prices
plt.plot(forecast_dates, forecast, label='Forecasted Prices', color='orange')

# Labels and title
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Forecast for Next 3 Months')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()