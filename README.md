# NVDA Stock Price Prediction Using LSTM

This repository contains a university machine learning assessment notebook titled
**Stock Price Prediction Using an LSTM-Based Recurrent Neural Network**.

The main notebook predicts the next-day NVIDIA (NVDA) closing price using
historical closing prices from Yahoo Finance for the period `2015-01-01` to
`2025-12-31`.

## Main File

- `Stock_Price_Prediction_LSTM_NVDA.ipynb`

## Dataset

- `NVDA_stock_data.csv`

The notebook downloads the data with `yfinance` and saves a local CSV copy. If a
fresh Yahoo Finance download is unavailable, the notebook can reuse the cached
CSV file.

## Methods

- Data collection from Yahoo Finance
- Closing-price preprocessing and MinMax normalization
- 60-day sequence preparation
- Chronological 80/20 train-test split
- LSTM recurrent neural network
- Random Forest comparison model
- MAE, RMSE, R2, and MAPE evaluation
- Result visualizations and critical analysis

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Notes

The notebook was tested end-to-end with TensorFlow on CPU. TensorFlow may show a
warning on native Windows that GPU support is unavailable; this does not stop the
notebook from running.
