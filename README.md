# Apple Stock Price Prediction Using an LSTM-Based Recurrent Neural Network

Course: **KIE4031 Machine Learning**

This repository contains a final summative assessment project that predicts the
next-day closing price of Apple Inc. stock using an LSTM-based Recurrent Neural
Network.

## Dataset

- Dataset source: **Yahoo Finance** using `yfinance`
- Stock ticker: `AAPL`
- Date range: `2015-01-01` to `2025-12-31`
- Saved dataset: `AAPL_stock_data.csv`

## Models

- Main model: **LSTM Recurrent Neural Network**
- Alternative model: **Random Forest Regressor**

The LSTM model uses a 60-day historical closing-price window to predict the next
day's closing price. The Random Forest model uses the same 60-day window flattened
into traditional machine learning features.

## How To Run

1. Clone the repository:

```bash
git clone https://github.com/BingJun69/ML_Final_Assesment.git
cd ML_Final_Assesment
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```text
stock_price_lstm_aapl.ipynb
```

4. Run all cells from top to bottom.

## Required Libraries

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `yfinance`
- `jupyter`

## Expected Outputs

The notebook displays:

- Dataset preview using head and tail tables
- Dataset shape, column names, and descriptive statistics
- Missing-value and duplicate-row checks
- Apple closing price trend graph
- Apple volume graph
- 50-day and 200-day moving average graph
- Training and validation loss graph
- LSTM actual vs predicted price graph
- Prediction error graph
- Sample prediction table
- LSTM vs Random Forest comparison table
- Actual price, LSTM prediction, and Random Forest prediction comparison graph
- Critical analysis and conclusion

## Notes

The notebook uses chronological train-test splitting and does not shuffle the
time-series data. The `MinMaxScaler` is fitted only on the training closing-price
data to avoid data leakage. This project is an educational forecasting study and
should not be treated as a real trading system.
