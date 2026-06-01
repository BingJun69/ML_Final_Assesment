# NVDA Stock Price Prediction with an LSTM RNN

## 1. Data Collection and Preprocessing

This project uses daily NVIDIA Corporation (NVDA) stock data from Yahoo Finance for
the period 2021-01-01 to 2026-04-30. The script downloads OHLCV data, sorts it
chronologically, forward-fills missing market values, removes remaining incomplete
rows, and engineers three extra variables:

- Daily return, calculated from closing price percentage change.
- 10-day moving average of close price.
- 20-day moving average of close price.

The final feature set is: Open, High, Low, Close, Volume, Return, MA_10, MA_20.
All features are scaled with `MinMaxScaler` fitted only on the training period to
avoid data leakage. The LSTM receives 60 trading days of historical features,
predicts the next-day return, and then reconstructs the predicted closing price
from the previous close. This return-based target is used because stock prices are
nonstationary and absolute-price extrapolation is unstable during strong trends.

Dataset size after cleaning: 1318 trading days.

Chronological split:

| Split | Rows | Date range |
| --- | ---: | --- |
| Train | 922 | 2021-02-01 to 2024-09-30 |
| Validation | 198 | 2024-10-01 to 2025-07-17 |
| Test | 198 | 2025-07-18 to 2026-04-30 |

## 2. Investigation of the Proposed Technique

The chosen RNN-based method is a Long Short-Term Memory (LSTM) neural network.
An LSTM is suitable for time-series prediction because it has recurrent connections
and gating mechanisms that help retain useful information across previous time
steps. Compared with a basic RNN, the input, forget, and output gates reduce the
vanishing-gradient problem and allow the model to learn longer temporal patterns.

For stock prediction, the LSTM can learn how recent price levels, volatility,
short-term trend, and volume interact over a sequence of trading days. However,
stock prices are noisy and affected by news, earnings, interest rates, and market
sentiment that are not fully represented in historical OHLCV data. Therefore, the
model should be treated as a forecasting experiment rather than a trading system.

## 3. Model Development and Evaluation

The LSTM model has two recurrent layers with dropout regularization:

- LSTM(64), return sequences enabled.
- Dropout(0.20).
- LSTM(32).
- Dropout(0.20).
- Dense(16, ReLU).
- Dense(1) output for next-day return.

The model uses the Adam optimizer, mean squared error loss, early stopping, and a
chronological validation set. It trained for 10 epochs.

An alternative model, Random Forest Regression, is used as a non-recurrent baseline.
It predicts close price using lagged close prices, lagged return, moving averages,
and volume. This comparison helps test whether the LSTM sequence model provides
value beyond a conventional machine-learning model using engineered lag features.

### Test Metrics

| Model | RMSE | MAE | MAPE (%) | R2 | Directional Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM RNN | 4.0078 | 3.1149 | 1.6878 | 0.7984 | 48.22 |
| Random Forest baseline | 18.3492 | 16.0698 | 8.5357 | -3.2265 | 41.12 |

The saved figures are:

- `outputs/01_price_history.png`
- `outputs/02_lstm_training_loss.png`
- `outputs/03_actual_vs_predicted.png`
- `outputs/04_lstm_residuals.png`
- `outputs/05_train_validation_test_split.png`
- `outputs/06_feature_correlation_heatmap.png`
- `outputs/07_model_error_comparison.png`
- `outputs/08_lstm_absolute_error_over_time.png`
- `outputs/09_daily_return_distribution.png`
- `outputs/10_workflow_diagram.png`

For a visual walkthrough, open `visual_summary.html`.

## 4. Critical Analysis

### Strengths

The LSTM uses sequential windows rather than isolated observations, so it can model
time-dependent patterns such as trend persistence and delayed effects from earlier
price movements. Feature scaling and chronological splitting make the training
process more stable and avoid look-ahead leakage. Early stopping limits unnecessary
training once validation performance stops improving.

### Limitations

The model depends heavily on historical price and volume data. This means it cannot
directly interpret new information such as product announcements, earnings surprises,
regulatory events, macroeconomic shocks, or market sentiment. NVDA also experienced
large AI-related price movements during this period, so the learned patterns may not
generalize well to calmer or structurally different market regimes.

Overfitting is another risk. LSTMs have many trainable parameters, while the dataset
contains only daily trading observations. Dropout and early stopping reduce this
risk, but they do not remove it. The training-loss plot should be reviewed for a
large gap between training and validation curves.

The model is also sensitive to market volatility. When prices move sharply, a model
trained on recent history can lag behind the actual market because it learns from
past observations rather than future catalysts.

### Comparison with the Alternative Model

The Random Forest baseline is easier to interpret and faster to train. It can capture
nonlinear relationships between lagged indicators, but it does not naturally model
ordered sequences in the way an LSTM does. If the LSTM performs better on RMSE,
MAE, or MAPE, it suggests the sequential representation adds useful information.
If the Random Forest performs similarly or better, then engineered lag features may
be sufficient for this dataset, and the added complexity of an LSTM may not be
justified.

## Source Code Link

Place your GitHub repository link here after uploading the project:

`https://github.com/<your-username>/<your-repository>`
