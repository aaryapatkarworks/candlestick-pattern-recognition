"""
Step 1: Download real Nifty 50 daily OHLCV data (ticker ^NSEI) from Yahoo Finance.
Run:  python 01_download_data.py
Output: nifty50.csv  (Date, Open, High, Low, Close, Volume)
Requires internet:  pip install yfinance
"""
import yfinance as yf

df = yf.download("^NSEI", start="2010-01-01", end="2025-12-31", auto_adjust=False, progress=False)
if isinstance(df.columns, __import__("pandas").MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
df = df[(df["High"] >= df["Low"]) & (df["Open"] > 0)]
df.index.name = "Date"
df.to_csv("nifty50.csv")
print(f"Saved nifty50.csv with {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
