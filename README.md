# TSLA Stock Price Prediction Using an LSTM-Based RNN

Course: **KIE4031 Machine Learning**

This project predicts Tesla Inc. (`TSLA`) next-trading-day closing price using an LSTM-based recurrent neural network and compares it with a Random Forest Regressor benchmark.

## Dataset

- Ticker: `TSLA`
- Data source: Yahoo Finance using `yfinance`
- Date range: past 10 years from the notebook run date
- Cached dataset: `TSLA_stock_data.csv`

## Models

- LSTM recurrent neural network
- Random Forest Regressor alternative model
- Previous-close baseline

## How To Run

```bash
pip install -r requirements.txt
```

Open and run:

```text
tsla_stock_price_rnn_prediction.ipynb
```

Run all cells from top to bottom.

Final report:

```text
TSLA_Stock_Price_Prediction_Report_Polished.docx
```

## Expected Outputs

- Yahoo Finance data download or cached CSV fallback
- Data quality checks and descriptive statistics
- TSLA close price, volume, and moving-average visualizations
- Engineered model features and chronological train-validation-test split
- LSTM model summary and training-loss plot
- Evaluation metrics: MAE, RMSE, R2, MAPE, and direction accuracy
- Comparison table for LSTM, Random Forest, and previous-close baseline
- Actual vs predicted test-period graph
- LSTM error distribution and absolute-error graph
- Investment simulation table and portfolio equity curve
- Latest next-day TSLA closing-price forecast

## Assessment Scope

The notebook covers the machine-learning implementation required by the rubric. Long-form discussion, critical analysis, and report writing can be added separately.
