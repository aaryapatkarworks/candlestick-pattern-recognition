"""
Generates a SYNTHETIC Nifty-like OHLC file (nifty50_sample.csv) ONLY to test that the code runs
offline. Do NOT use it for the research paper -- use 01_download_data.py for real data.
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 3000
dates = pd.bdate_range("2012-01-02", periods=n)
ret = rng.normal(0.0004, 0.01, n)
close = 5000 * np.exp(np.cumsum(ret))
open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
vol = rng.integers(1e5, 5e5, n)
pd.DataFrame({"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}) \
  .to_csv("nifty50_sample.csv", index=False)
print("Saved nifty50_sample.csv (synthetic)")
