import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Download stock data
stock = "AAPL"  # Apple stock
data = yf.download(stock, start="2020-01-01", end="2025-01-01")

# Keep only Close price
data = data[['Close']]

# Create target column (next day's close price)
data['Target'] = data['Close'].shift(-1)

# Remove last row with NaN target
data.dropna(inplace=True)

# Features and target
X = data[['Close']]
y = data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title(f'{stock} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# Predict next day price
last_close = data[['Close']].iloc[-1]
next_day_price = model.predict([last_close])[0]

print(f"\nPredicted Next Day Closing Price: ${next_day_price:.2f}")