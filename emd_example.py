import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from datetime import datetime, timedelta

# Download Bitcoin data
ticker = "BTC-USD"
end_date = datetime.now()
start_date = end_date - timedelta(days=180)  # 6 months

btc = yf.download(ticker, start=start_date, end=end_date, progress=False)
prices = btc['Close'].values

# Fill any NaN values
prices = np.nan_to_num(prices, nan=np.nanmean(prices))

# Calculate log returns for stationarity
returns = np.diff(np.log(prices))

# Use last 500 returns for EMD
signal = returns[-500:] if len(returns) > 500 else returns

# EMD decomposition
emd = EMD()
imfs = emd(signal)

print(f"Signal shape: {signal.shape}")
print(f"IMFs shape: {imfs.shape}")
print(f"Number of IMFs extracted: {len(imfs)}")

# Key insight: EMD automatically separates scales
print(f"\nVariance distribution across IMFs (scale separation):")
for i in range(min(5, len(imfs))):  # First 5 IMFs
    imf_var = np.var(imfs[i])
    total_var = np.var(signal)
    percentage = 100 * imf_var / total_var if total_var > 0 else 0
    print(f"  IMF {i+1}: variance = {imf_var:.6f} ({percentage:.1f}% of total)")

# The last IMF contains the residual trend component
if len(imfs) > 0:
    print(f"\nLast IMF (residual trend) variance: {np.var(imfs[-1]):.6f}")
    print("Note: In return space, this 'trend' represents slowly-changing")
    print("      volatility or market regime shifts, not price trends.")

# Visualize IMFs
fig, axes = plt.subplots(imfs.shape[0] + 2, 1, figsize=(14, 3*imfs.shape[0]))
axes[0].plot(signal)
axes[0].set_title('Bitcoin Daily Returns (Original Signal)')
axes[0].set_ylabel('Log Return')
axes[0].grid(True, alpha=0.3)

# Plot each IMF with variance annotation
for i in range(imfs.shape[0]):
    axes[i+1].plot(imfs[i])
    imf_var = np.var(imfs[i])
    axes[i+1].set_title(f'IMF {i+1} (var = {imf_var:.6f})')
    axes[i+1].set_ylabel(f'IMF {i+1}')
    axes[i+1].grid(True, alpha=0.3)

# Plot residual (trend component)
if len(imfs) > 0:
    axes[-1].plot(imfs[-1])
else:
    axes[-1].plot(signal)
axes[-1].set_title('Residual (Slowest Trend Component)')
axes[-1].set_ylabel('Residual')
axes[-1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Verify perfect reconstruction (important property)
if len(imfs) > 0:
    reconstructed = np.sum(imfs, axis=0)
    reconstruction_error = np.mean(np.abs(signal - reconstructed))
    print(f"\nReconstruction error (MAE): {reconstruction_error:.6f}")
    print("Expected: ~0 (perfect reconstruction is a key feature of EMD)")