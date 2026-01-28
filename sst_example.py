import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import ssq_cwt, cwt, icwt
from ssqueezepy.visuals import imshow
from datetime import datetime, timedelta

# Download VIX volatility index data
ticker = "^VIX"
end_date = datetime.now()
start_date = end_date - timedelta(days=180)  # 6 months

vix = yf.download(ticker, start=start_date, end=end_date, progress=False)
prices = vix['Close'].values

# Fill any NaN values
prices = np.nan_to_num(prices, nan=np.nanmean(prices))

# Use normalized VIX for better visualization
signal = (prices - np.mean(prices)) / np.std(prices)
signal = signal[-256:] if len(signal) > 256 else signal  # Use last 256 points

# Synchrosqueezing Transform
Tx, Wx, ssq_freqs, scales = ssq_cwt(signal)

# Traditional CWT for comparison
Wx_cwt, scales_cwt = cwt(signal, 'morlet')

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Original signal
axes[0].plot(signal)
axes[0].set_title('VIX Volatility Index (Normalized)')
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Normalized VIX')
axes[0].grid(True, alpha=0.3)

# Traditional CWT
im = axes[1].imshow(np.abs(Wx_cwt), aspect='auto', 
                   extent=[0, len(signal), scales_cwt[-1], scales_cwt[0]],
                   cmap='viridis')
axes[1].set_title('Traditional Continuous Wavelet Transform')
axes[1].set_xlabel('Time (days)')
axes[1].set_ylabel('Scale')
plt.colorbar(im, ax=axes[1], label='Magnitude')

# Synchrosqueezed Transform
im = axes[2].imshow(np.abs(Tx), aspect='auto',
                   extent=[0, len(signal), ssq_freqs[-1], ssq_freqs[0]],
                   cmap='plasma')
axes[2].set_title('Synchrosqueezed Transform (Sharpened)')
axes[2].set_xlabel('Time (days)')
axes[2].set_ylabel('Frequency')
plt.colorbar(im, ax=axes[2], label='Magnitude')

plt.tight_layout()
plt.show()

print(f"Signal length: {len(signal)}")
print(f"SST shape: {Tx.shape}")
print(f"Frequency range: [{ssq_freqs[0]:.3f}, {ssq_freqs[-1]:.3f}] Hz")
print(f"Maximum energy at time point: {np.argmax(np.sum(np.abs(Tx), axis=0))}")