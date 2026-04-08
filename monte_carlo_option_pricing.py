import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import sys

# For reproducibility
np.random.seed(42)

# --------------------------
# 1. Get real market data with error handling
# --------------------------
ticker = 'AAPL'
print(f"Downloading data for {ticker}...")
try:
    data = yf.download(ticker, period='1y', interval='1d', progress=False)
except Exception as e:
    print(f"Download failed with exception: {e}")
    sys.exit(1)

# Check if data is empty
if data.empty:
    print("No data returned. Please check ticker symbol or internet connection.")
    sys.exit(1)

# Flatten columns if they are a MultiIndex (common in newer yfinance versions)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Select the best available price column
if 'Adj Close' in data.columns:
    prices = data['Adj Close']
elif 'Close' in data.columns:
    prices = data['Close']
else:
    print("Neither 'Adj Close' nor 'Close' found in data. Available columns:", list(data.columns))
    sys.exit(1)

# Drop any NaNs from prices
prices = prices.dropna()

if prices.empty:
    print("Price data is empty after dropping NaNs.")
    sys.exit(1)

log_returns = np.log(prices / prices.shift(1)).dropna()
sigma = log_returns.std() * np.sqrt(252)   # historical volatility (annual)
S0 = prices.iloc[-1]                        # current stock price

print(f"Data sample (last 5 rows):\n{data.tail()}")
print(f"Ticker: {ticker}")
print(f"Current price: {S0:.2f}")
print(f"Historical volatility (annual): {sigma:.4f}")

# --------------------------
# 2. Fixed parameters
# --------------------------
K = S0 * 1.05      # strike 5% above current price (ATM +5%)
T = 1.0            # 1 year
r = 0.05           # risk-free rate (simplified; could fetch current rate)
n_sims = 10000
n_steps = 252

dt = T / n_steps
drift = (r - 0.5 * sigma**2) * dt
vol = sigma * np.sqrt(dt)

# --------------------------
# 3. Monte Carlo simulation
# --------------------------
prices_sim = np.zeros((n_steps + 1, n_sims))
prices_sim[0] = S0

for i in range(1, n_steps + 1):
    z = np.random.standard_normal(n_sims)
    prices_sim[i] = prices_sim[i-1] * np.exp(drift + vol * z)

final_prices = prices_sim[-1]
payoffs = np.maximum(final_prices - K, 0)
option_price = np.exp(-r * T) * np.mean(payoffs)

# Standard error and confidence interval for the price
stderr = np.std(payoffs) / np.sqrt(n_sims)
ci_lower = option_price - 1.96 * stderr
ci_upper = option_price + 1.96 * stderr

print(f"\nMonte Carlo call price: ${option_price:.2f}")
print(f"95% CI: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# --------------------------
# 4. Black‑Scholes comparison
# --------------------------
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

bs_price = black_scholes_call(S0, K, T, r, sigma)
print(f"Black‑Scholes price:   ${bs_price:.2f}")
print(f"Difference:            ${abs(bs_price - option_price):.4f}")

# --------------------------
# 5. Risk measures (VaR and CVaR) from option P&L at maturity
# --------------------------
# Future value of the option premium
future_premium = option_price * np.exp(r * T)

# P&L at maturity (future dollars)
pnl = payoffs - future_premium

confidence = 0.95
var = np.percentile(pnl, (1 - confidence) * 100)   # VaR at 95% (loss not exceeded with 95% confidence)
cvar = pnl[pnl <= var].mean()                       # CVaR: average loss beyond VaR

print(f"\n--- Risk Measures (at {confidence:.0%} confidence) ---")
print(f"VaR (Value at Risk):      ${var:.2f}")
print(f"CVaR (Conditional VaR):   ${cvar:.2f}")

# Plot histogram of P&L with VaR and CVaR
plt.figure(figsize=(10, 5))
plt.hist(pnl, bins=50, edgecolor='black', alpha=0.7, density=True)
plt.axvline(var, color='red', linestyle='--', linewidth=2, label=f'VaR ({confidence:.0%}) = ${var:.2f}')
plt.axvline(cvar, color='darkred', linestyle=':', linewidth=2, label=f'CVaR = ${cvar:.2f}')
plt.xlabel('Profit / Loss at maturity ($)')
plt.ylabel('Density')
plt.title(f'Distribution of Option P&L at Maturity\n{ticker} Call, Strike=${K:.2f}, Option Price=${option_price:.2f}')
plt.legend()
plt.grid(True)
plt.show()

# Optional: also show final price distribution with VaR/CVaR on price
plt.figure(figsize=(10, 5))
plt.hist(final_prices, bins=50, edgecolor='black', alpha=0.7, density=True)
plt.axvline(np.percentile(final_prices, (1 - confidence) * 100), color='red', linestyle='--', label=f'Price VaR ({confidence:.0%})')
plt.axvline(S0, color='green', linestyle='-', label=f'S0 = ${S0:.2f}')
plt.xlabel('Final stock price ($)')
plt.ylabel('Density')
plt.title(f'Distribution of Final Stock Price')
plt.legend()
plt.grid(True)
plt.show()