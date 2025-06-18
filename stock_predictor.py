import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Download Data
df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
df = df[['Close']]
df.dropna(inplace=True)

# 2. Create Target Column (Next Day's Close)
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# 3. Split Data
X = df[['Close']]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 5. Evaluate
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse:.4f}')

# 6. Plot Results
plt.figure(figsize=(14,5))
plt.plot(y_test.index, y_test, label='Actual Price')
plt.plot(y_test.index, predictions, label='Predicted Price')
plt.legend()
plt.title('Stock Price Prediction (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()
