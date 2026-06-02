# Apple Stock Price Prediction Using an LSTM-Based Recurrent Neural Network

Course: **KIE4031 Machine Learning**

This project predicts Apple Inc. (`AAPL`) closing stock price using a simple LSTM model and compares it with a Random Forest Regressor.

## Dataset

- Ticker: `AAPL`
- Data source: Yahoo Finance using `yfinance`
- Date range: `2015-01-01` to `2026-01-01`
- Saved dataset: `AAPL_stock_data.csv`

## Models

- LSTM Recurrent Neural Network
- Random Forest Regressor

## How To Run

```bash
pip install -r requirements.txt
```

Open and run:

```text
stock_price_lstm_aapl.ipynb
```

Run all cells from top to bottom.

## Expected Outputs

- First and last rows of the dataset
- Dataset shape, columns, and descriptive statistics
- Missing-value and duplicate-row checks
- AAPL close price graph
- AAPL volume graph
- Close price with 50-day and 200-day moving averages
- LSTM model summary
- Training and validation loss graph
- LSTM evaluation metrics
- LSTM vs Random Forest comparison table
- Actual vs predicted price graphs
- Sample prediction table

## Requirements

See `requirements.txt`.
