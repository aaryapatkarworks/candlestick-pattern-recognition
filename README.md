# Candlestick Pattern Recognition and Price Movement Prediction for Nifty 50 Index

**Course:** Pattern Recognition (AY 2026-27, Sem-7)  
**Group No:** 16  
**Group Members:**
* Aarya Patkar (1032232017)
* Pranav More (1032232122)
* Aaditya Temgire (1032232232)
* Sujal Shirole (1032233387)

---

## Project Overview
This project implements and compares three distinct pattern recognition paradigms to predict 5-day-ahead price movement directions (`Up` / `Down`) for the Nifty 50 Index using historical daily OHLC data.

### Implemented Solutions:
1. **Solution 1:** Rule-based candlestick pattern features + **Filter Feature Selection (SelectKBest / ANOVA)** + $k$-Nearest Neighbors ($k$-NN).
2. **Solution 2:** Rendered candlestick chart images + **HOG (Histogram of Oriented Gradients)** + **Feature Selection** + Support Vector Machine (SVM).
3. **Solution 3:** Rendered candlestick chart images + End-to-End Convolutional Neural Network (CNN) with embedded regularization.

---

## Repository Structure
```text
CCA1/
├── 01_download_data.py          # Script to download and prepare Nifty 50 dataset
├── candlestick_solutions.py     # Main script executing S1, S2, S3 + Feature Selection
├── make_sample_data.py          # Script for generating synthetic smoke test data
├── nifty50.csv                  # Actual historical dataset (or nifty50_sample.csv)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
