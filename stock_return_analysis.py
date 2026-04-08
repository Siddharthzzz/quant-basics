import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Download stock data
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Show first few rows
print(data.head(10))

data["Simple Return"] = data["Close"].pct_change()

data["log Return"] = np.log(data["Close"] / data["Close"].shift(1))
data = data.dropna()

plt.figure(figsize=(10,5))
plt.plot(data["Simple Return"])
plt.title("Daily Simple Returns")
plt.show()

mean_return = data["Simple Return"].mean()
Variance = data["Simple Return"].var()
volatility = data["Simple Return"].std()

print("Average Daily Return:", mean_return)
print("Variance",Variance)
print("Daily Volatility:", volatility)

annual_volatility = volatility * np.sqrt(252)
print("Annual Volatility:", annual_volatility)

data['Cumulative Return'] = (1 + data['Simple Return']).cumprod()

plt.figure(figsize=(10,5))
plt.plot(data['Cumulative Return'])
plt.title("Cumulative Returns")
plt.show()

