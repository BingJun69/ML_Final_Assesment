# Stock Price Prediction Using an LSTM-Based Recurrent Neural Network

Course: **KIE4031 Machine Learning**

This repository contains a final summative assessment project that predicts the
next-day closing price of NVIDIA Corporation (NVDA) stock using an LSTM-based
Recurrent Neural Network.

## Dataset

The dataset is collected from **Yahoo Finance** using the `yfinance` Python
library.

- Ticker: `NVDA`
- Date range: `2015-01-01` to `2025-12-31`
- Saved dataset: `NVDA_stock_data.csv`

The notebook downloads fresh data when available and can reuse the local CSV file
if Yahoo Finance is temporarily unavailable.

## Models

- Main model: **LSTM Recurrent Neural Network**
- Alternative model: **Random Forest Regressor**

The LSTM model uses a 60-day historical closing-price window to predict the next
day's closing price. The Random Forest model uses the same 60-day window, flattened
into traditional machine learning features, for comparison.

## Main Notebook

Run this notebook from top to bottom:

```text
stock_price_lstm_nvda.ipynb
```

## Required Libraries

Install the required packages with:

```bash
pip install -r requirements.txt
```

Required libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `yfinance`
- `jupyter`

## Expected Outputs and Graphs

The notebook produces:

- Downloaded NVDA stock dataset
- Missing-value and duplicate-row checks
- Original closing-price trend graph
- LSTM training and validation loss graph
- Actual vs LSTM predicted closing-price graph
- LSTM prediction error over time graph
- Sample prediction table with absolute and percentage error
- Random Forest comparison metrics table
- Actual vs predicted graph comparing LSTM and Random Forest
- Critical analysis and final conclusion

The notebook also saves selected graphs to the `outputs/` folder when run.

## Notes

The time-series data is split chronologically and is not shuffled. The
`MinMaxScaler` is fitted only on the training-period data to avoid future data
leakage. TensorFlow may show a warning on native Windows that GPU support is not
available; the notebook still runs correctly on CPU.
