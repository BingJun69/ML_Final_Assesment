# Apple Stock Price Prediction Using an LSTM-Based Recurrent Neural Network

Course: **KIE4031 Machine Learning**

This repository contains a final summative assessment project that predicts the
next-day closing price of Apple Inc. stock using an LSTM-based Recurrent Neural
Network. The notebook also tests whether adding OHLCV features, technical
indicators, and a return-based prediction target improves accuracy compared with
a close-only price-target LSTM baseline.

## Dataset

- Dataset source: **Yahoo Finance** using `yfinance`
- Stock ticker: `AAPL`
- Date range: `2015-01-01` to `2025-12-31`
- Saved dataset: `AAPL_stock_data.csv`

## Models

- Baseline model: **Close-only price-target LSTM Recurrent Neural Network**
- Improved model: **Enhanced return-target LSTM with OHLCV features and technical indicators**
- Alternative model: **Random Forest return model**

The enhanced LSTM model uses a 60-day historical window with Open, High, Low,
Close, Volume, returns, moving averages, RSI, MACD, Bollinger Bands, and rolling
volatility. It predicts next-day return and reconstructs the next-day closing
price. The Random Forest model uses the same enhanced 60-day window and return
target flattened into traditional machine learning features.

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
- Technical-indicator feature table
- Close-only price LSTM vs enhanced return LSTM validation loss graph
- Close-only price LSTM vs enhanced return LSTM metric table
- Actual vs predicted price graph
- Prediction error graph
- Sample prediction table
- Close-only price LSTM vs enhanced return LSTM vs Random Forest comparison table
- Actual price, enhanced return LSTM prediction, and Random Forest prediction comparison graph
- Critical analysis and conclusion

## Notes

The notebook uses chronological train-test splitting and does not shuffle the
time-series data. The `MinMaxScaler` is fitted only on the training closing-price
data to avoid data leakage. This project is an educational forecasting study and
should not be treated as a real trading system.
