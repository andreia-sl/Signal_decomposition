import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN
from datetime import datetime, timedelta

# Download NASDAQ data
ticker = "^IXIC"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)  # 1 year

nasdaq = yf.download(ticker, start=start_date, end=end_date, progress=False)
prices = nasdaq['Close'].values

# Fill any NaN values
prices = np.nan_to_num(prices, nan=np.nanmean(prices))

# Calculate log returns
returns = np.diff(np.log(prices))

# Use last 400 returns for decomposition
signal = returns[-400:] if len(returns) > 400 else returns

# CEEMDAN decomposition
ceemdan = CEEMDAN()
imfs = ceemdan(signal)

print(f"Signal length: {len(signal)}")
print(f"IMFs extracted: {imfs.shape[0]}")

# Plot decomposition
fig, axes = plt.subplots(imfs.shape[0] + 1, 1, figsize=(12, 2.5*(imfs.shape[0]+1)))

axes[0].plot(signal)
axes[0].set_title('NASDAQ Daily Returns (Original)')
axes[0].set_ylabel('Returns')
axes[0].grid(True, alpha=0.3)

for i in range(imfs.shape[0]):
    axes[i+1].plot(imfs[i])
    axes[i+1].set_title(f'CEEMDAN IMF {i+1}')
    axes[i+1].set_ylabel(f'IMF {i+1}')
    axes[i+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analyse IMF frequencies using zero-crossing method
print("\nIMF Characteristics:")
for i in range(min(5, imfs.shape[0])):  # First 5 IMFs
    zero_crossings = np.where(np.diff(np.sign(imfs[i])))[0]
    if len(zero_crossings) > 1:
        avg_period = 2 * len(imfs[i]) / len(zero_crossings)
        print(f"IMF {i+1}: {len(zero_crossings)} zero crossings, "
              f"approx period: {avg_period:.1f} days, "
              f"variance: {np.var(imfs[i]):.6f}")