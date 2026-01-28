import yfinance as yf
import pywt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Download S&P 500 data
ticker = "^GSPC"
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # 2 years of data

sp500 = yf.download(ticker, start=start_date, end=end_date)
prices = sp500['Close'].values

# Fill any NaN values (weekends/holidays)
prices = np.nan_to_num(prices, nan=np.nanmean(prices))

# Calculate daily returns (more stationary than prices)
returns = np.diff(np.log(prices))

# Use last 1024 returns for decomposition (power of 2 for clean wavelet decomposition)
signal = returns[-1024:] if len(returns) > 1024 else returns

# Wavelet decomposition
coeffs = pywt.wavedec(signal, wavelet='db4', level=5)

# Visualize
fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(12, 10))
axes[0].plot(signal)
axes[0].set_title(f'S&P 500 Daily Returns (Last {len(signal)} days)')
axes[0].set_ylabel('Log Return')

for i, coeff in enumerate(coeffs):
    axes[i+1].plot(coeff)
    if i == 0:
        axes[i+1].set_title(f'Wavelet Approximation (Level {i})')
    else:
        axes[i+1].set_title(f'Wavelet Detail (Level {i})')
    axes[i+1].set_ylabel(f'Scale {i}')

plt.tight_layout()
plt.show()

print(f"Signal length: {len(signal)}")
print(f"Number of wavelet coefficients: {[len(c) for c in coeffs]}")
print(f"Total variance in original: {np.var(signal):.6f}")
print(f"Variance in approximation: {np.var(coeffs[0]):.6f} ({100*np.var(coeffs[0])/np.var(signal):.1f}%)")