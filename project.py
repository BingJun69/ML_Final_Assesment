import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential


TICKER = "NVDA"
START_DATE = "2021-01-01"
END_DATE = "2026-05-01"  # yfinance end date is exclusive; this includes April 2026.
LOOKBACK_DAYS = 60
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Return", "MA_10", "MA_20"]
TARGET_COLUMN = "Close"
RETURN_TARGET_COLUMN = "Return"
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
RANDOM_SEED = 42


def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] else col[-1] for col in df.columns]
    return df


def download_nvda_data(start: str, end: str, force_download: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    raw_path = DATA_DIR / "NVDA_2021_to_2026_april_raw.csv"

    if raw_path.exists() and not force_download:
        df = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
    else:
        df = yf.download(TICKER, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise RuntimeError("No data was downloaded. Check the ticker, dates, or internet connection.")
        df = flatten_yfinance_columns(df)
        df.to_csv(raw_path, index_label="Date")

    return df


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().dropna()

    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df = df.dropna()

    cleaned_path = DATA_DIR / "NVDA_2021_to_2026_april_cleaned.csv"
    df.to_csv(cleaned_path, index_label="Date")
    return df


def make_sequences(
    feature_values: np.ndarray, target_values: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(lookback, len(feature_values)):
        x.append(feature_values[i - lookback : i])
        y.append(target_values[i])
    return np.array(x), np.array(y)


def reconstruct_close_from_return(previous_close: pd.Series, predicted_return: np.ndarray) -> np.ndarray:
    return previous_close.to_numpy() * (1.0 + np.asarray(predicted_return).reshape(-1))


def chronological_split(
    df: pd.DataFrame, train_ratio: float, validation_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def build_lstm_model(lookback: int, n_features: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(lookback, n_features)),
            LSTM(64, return_sequences=True),
            Dropout(0.20),
            LSTM(32),
            Dropout(0.20),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def make_baseline_features(df: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(index=df.index)
    for lag in [1, 2, 3, 5, 10, 20, lookback]:
        features[f"close_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)
    features["return_lag_1"] = df["Return"].shift(1)
    features["ma_10_lag_1"] = df["MA_10"].shift(1)
    features["ma_20_lag_1"] = df["MA_20"].shift(1)
    features["volume_lag_1"] = df["Volume"].shift(1)

    dataset = features.join(df[TARGET_COLUMN].rename("target")).dropna()
    return dataset.drop(columns=["target"]), dataset["target"]


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
    r2 = float(r2_score(actual, predicted))

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predicted))
    directional_accuracy = float(np.mean(actual_direction == predicted_direction) * 100)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_percent": mape,
        "R2": r2,
        "Directional_Accuracy_percent": directional_accuracy,
    }


def save_plots(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    history: tf.keras.callbacks.History,
    result_df: pd.DataFrame,
    lstm_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df["Close"], label="NVDA close", color="#1f77b4")
    plt.plot(df.index, df["MA_20"], label="20-day moving average", color="#ff7f0e", alpha=0.85)
    plt.title("NVDA Closing Price, 2021 to April 2026")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_price_history.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title("LSTM Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_lstm_training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(result_df["Date"], result_df["Actual_Close"], label="actual", color="#222222")
    plt.plot(result_df["Date"], result_df["LSTM_Predicted_Close"], label="LSTM", color="#2ca02c")
    plt.plot(
        result_df["Date"],
        result_df["RandomForest_Predicted_Close"],
        label="Random forest baseline",
        color="#d62728",
        alpha=0.85,
    )
    plt.title("Actual vs Predicted NVDA Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_actual_vs_predicted.png", dpi=160)
    plt.close()

    residuals = result_df["Actual_Close"] - result_df["LSTM_Predicted_Close"]
    plt.figure(figsize=(10, 5))
    plt.scatter(result_df["LSTM_Predicted_Close"], residuals, alpha=0.7, color="#9467bd")
    plt.axhline(0, color="#222222", linewidth=1)
    plt.title("LSTM Residuals")
    plt.xlabel("Predicted close (USD)")
    plt.ylabel("Actual - predicted (USD)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_lstm_residuals.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.plot(train_df.index, train_df["Close"], label="train", color="#1f77b4")
    plt.plot(val_df.index, val_df["Close"], label="validation", color="#ff7f0e")
    plt.plot(test_df.index, test_df["Close"], label="test", color="#2ca02c")
    plt.axvline(val_df.index.min(), color="#444444", linestyle="--", linewidth=1)
    plt.axvline(test_df.index.min(), color="#444444", linestyle="--", linewidth=1)
    plt.title("Chronological Train, Validation, and Test Split")
    plt.xlabel("Date")
    plt.ylabel("Close price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_train_validation_test_split.png", dpi=160)
    plt.close()

    corr = df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(8, 6))
    image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Correlation")
    plt.xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45, ha="right")
    plt.yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_feature_correlation_heatmap.png", dpi=160)
    plt.close()

    metric_names = ["RMSE", "MAE", "MAPE_percent"]
    display_names = ["RMSE", "MAE", "MAPE (%)"]
    x = np.arange(len(metric_names))
    width = 0.36
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, [lstm_metrics[m] for m in metric_names], width, label="LSTM", color="#2ca02c")
    plt.bar(
        x + width / 2,
        [baseline_metrics[m] for m in metric_names],
        width,
        label="Random forest",
        color="#d62728",
    )
    plt.xticks(x, display_names)
    plt.title("Model Error Comparison")
    plt.ylabel("Lower is better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_model_error_comparison.png", dpi=160)
    plt.close()

    result_with_errors = result_df.copy()
    result_with_errors["Absolute_Error"] = (
        result_with_errors["Actual_Close"] - result_with_errors["LSTM_Predicted_Close"]
    ).abs()
    plt.figure(figsize=(11, 5))
    plt.plot(result_with_errors["Date"], result_with_errors["Absolute_Error"], color="#9467bd")
    plt.title("LSTM Absolute Error Over the Test Period")
    plt.xlabel("Date")
    plt.ylabel("Absolute error (USD)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_lstm_absolute_error_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.hist(df["Return"] * 100, bins=60, color="#17becf", edgecolor="#222222", alpha=0.85)
    plt.axvline(0, color="#222222", linewidth=1)
    plt.title("Distribution of Daily NVDA Returns")
    plt.xlabel("Daily return (%)")
    plt.ylabel("Trading days")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "09_daily_return_distribution.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    steps = [
        "Yahoo Finance\nNVDA OHLCV",
        "Clean data\nAdd returns + MAs",
        "Scale features\nBuild 60-day windows",
        "Train LSTM\nPredict return",
        "Reconstruct close\nEvaluate vs baseline",
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    for index, (x_pos, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_pos,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f2f6ff", edgecolor="#2f5597"),
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.075, 0.55),
                xytext=(x_pos + 0.075, 0.55),
                arrowprops=dict(arrowstyle="->", color="#444444", linewidth=1.5),
            )
    ax.set_title("Assessment Workflow", fontsize=14, pad=18)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_workflow_diagram.png", dpi=160)
    plt.close()


def write_visual_summary(lstm_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> None:
    cards = [
        ("01_price_history.png", "Raw price movement and 20-day trend"),
        ("05_train_validation_test_split.png", "How the data is split without shuffling"),
        ("06_feature_correlation_heatmap.png", "How the engineered features relate to each other"),
        ("10_workflow_diagram.png", "The complete machine-learning workflow"),
        ("02_lstm_training_loss.png", "Whether the LSTM is learning or overfitting"),
        ("03_actual_vs_predicted.png", "Actual close price against model predictions"),
        ("07_model_error_comparison.png", "LSTM compared with the Random Forest baseline"),
        ("08_lstm_absolute_error_over_time.png", "Where prediction errors are largest"),
        ("04_lstm_residuals.png", "Prediction residual pattern"),
        ("09_daily_return_distribution.png", "Daily return volatility in the dataset"),
    ]
    card_html = "\n".join(
        f"""
        <section class="panel">
          <h2>{title}</h2>
          <img src="outputs/{filename}" alt="{title}">
        </section>
        """
        for filename, title in cards
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NVDA LSTM Visual Summary</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    header {{
      padding: 28px 40px 16px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee8;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      padding: 18px 40px;
      background: #ffffff;
      border-bottom: 1px solid #d8dee8;
    }}
    .metric {{
      padding: 12px 14px;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      background: #fbfcfe;
    }}
    .metric strong {{
      display: block;
      font-size: 13px;
      color: #52606d;
      margin-bottom: 5px;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      padding: 24px 40px 40px;
    }}
    .panel {{
      background: #ffffff;
      border: 1px solid #d8dee8;
      border-radius: 8px;
      padding: 14px;
    }}
    h2 {{
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border: 1px solid #edf0f5;
      border-radius: 6px;
      background: #ffffff;
    }}
    @media (max-width: 640px) {{
      header, .metrics, main {{
        padding-left: 16px;
        padding-right: 16px;
      }}
      main {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>NVDA LSTM Stock Prediction Visual Summary</h1>
    <p>Visual guide for the assessment: data, preprocessing, model behavior, comparison, and errors.</p>
  </header>
  <div class="metrics">
    <div class="metric"><strong>LSTM RMSE</strong>{lstm_metrics["RMSE"]:.4f}</div>
    <div class="metric"><strong>LSTM MAPE</strong>{lstm_metrics["MAPE_percent"]:.4f}%</div>
    <div class="metric"><strong>LSTM R2</strong>{lstm_metrics["R2"]:.4f}</div>
    <div class="metric"><strong>Baseline RMSE</strong>{baseline_metrics["RMSE"]:.4f}</div>
  </div>
  <main>
    {card_html}
  </main>
</body>
</html>
"""
    Path("visual_summary.html").write_text(html, encoding="utf-8")


def write_report(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lstm_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    epochs_trained: int,
) -> None:
    report = f"""# NVDA Stock Price Prediction with an LSTM RNN

## 1. Data Collection and Preprocessing

This project uses daily NVIDIA Corporation (NVDA) stock data from Yahoo Finance for
the period 2021-01-01 to 2026-04-30. The script downloads OHLCV data, sorts it
chronologically, forward-fills missing market values, removes remaining incomplete
rows, and engineers three extra variables:

- Daily return, calculated from closing price percentage change.
- 10-day moving average of close price.
- 20-day moving average of close price.

The final feature set is: {", ".join(FEATURE_COLUMNS)}.
All features are scaled with `MinMaxScaler` fitted only on the training period to
avoid data leakage. The LSTM receives 60 trading days of historical features,
predicts the next-day return, and then reconstructs the predicted closing price
from the previous close. This return-based target is used because stock prices are
nonstationary and absolute-price extrapolation is unstable during strong trends.

Dataset size after cleaning: {len(df)} trading days.

Chronological split:

| Split | Rows | Date range |
| --- | ---: | --- |
| Train | {len(train_df)} | {train_df.index.min().date()} to {train_df.index.max().date()} |
| Validation | {len(val_df)} | {val_df.index.min().date()} to {val_df.index.max().date()} |
| Test | {len(test_df)} | {test_df.index.min().date()} to {test_df.index.max().date()} |

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
chronological validation set. It trained for {epochs_trained} epochs.

An alternative model, Random Forest Regression, is used as a non-recurrent baseline.
It predicts close price using lagged close prices, lagged return, moving averages,
and volume. This comparison helps test whether the LSTM sequence model provides
value beyond a conventional machine-learning model using engineered lag features.

### Test Metrics

| Model | RMSE | MAE | MAPE (%) | R2 | Directional Accuracy (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM RNN | {lstm_metrics["RMSE"]:.4f} | {lstm_metrics["MAE"]:.4f} | {lstm_metrics["MAPE_percent"]:.4f} | {lstm_metrics["R2"]:.4f} | {lstm_metrics["Directional_Accuracy_percent"]:.2f} |
| Random Forest baseline | {baseline_metrics["RMSE"]:.4f} | {baseline_metrics["MAE"]:.4f} | {baseline_metrics["MAPE_percent"]:.4f} | {baseline_metrics["R2"]:.4f} | {baseline_metrics["Directional_Accuracy_percent"]:.2f} |

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
"""
    Path("report.md").write_text(report, encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    set_reproducibility()
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_df = download_nvda_data(args.start, args.end, force_download=args.force_download)
    df = clean_and_engineer_features(raw_df)

    train_df, val_df, test_df = chronological_split(df, args.train_ratio, args.validation_ratio)
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = feature_scaler.fit_transform(train_df[FEATURE_COLUMNS])
    train_target_scaled = target_scaler.fit_transform(train_df[[RETURN_TARGET_COLUMN]]).reshape(-1)

    # Include the previous lookback days before each split so validation/test windows have context.
    val_context = pd.concat([train_df.tail(args.lookback), val_df])
    test_context = pd.concat([val_df.tail(args.lookback), test_df])
    val_context_scaled = feature_scaler.transform(val_context[FEATURE_COLUMNS])
    test_context_scaled = feature_scaler.transform(test_context[FEATURE_COLUMNS])
    val_target_scaled = target_scaler.transform(val_context[[RETURN_TARGET_COLUMN]]).reshape(-1)
    test_target_scaled = target_scaler.transform(test_context[[RETURN_TARGET_COLUMN]]).reshape(-1)

    x_train, y_train = make_sequences(train_scaled, train_target_scaled, args.lookback)
    x_val, y_val = make_sequences(val_context_scaled, val_target_scaled, args.lookback)
    x_test, y_test = make_sequences(test_context_scaled, test_target_scaled, args.lookback)

    model = build_lstm_model(args.lookback, len(FEATURE_COLUMNS))
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=False,
    )

    lstm_pred_scaled = model.predict(x_test, verbose=0).reshape(-1)
    lstm_pred_return = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).reshape(-1)
    actual_close = test_df[TARGET_COLUMN].to_numpy()
    previous_close = test_context[TARGET_COLUMN].iloc[args.lookback - 1 : -1]
    lstm_pred_close = reconstruct_close_from_return(previous_close, lstm_pred_return)

    baseline_x, baseline_y = make_baseline_features(df, args.lookback)
    baseline_train_end = train_df.index.max()
    baseline_val_end = val_df.index.max()
    baseline_train_mask = baseline_x.index <= baseline_val_end
    baseline_test_mask = baseline_x.index > baseline_val_end

    baseline_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    baseline_model.fit(baseline_x.loc[baseline_train_mask], baseline_y.loc[baseline_train_mask])
    baseline_pred_close = baseline_model.predict(baseline_x.loc[baseline_test_mask])

    test_dates = test_df.index
    result_df = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual_Close": actual_close,
            "LSTM_Predicted_Close": lstm_pred_close,
            "RandomForest_Predicted_Close": baseline_pred_close[: len(test_dates)],
        }
    )
    result_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    lstm_metrics = calculate_metrics(result_df["Actual_Close"], result_df["LSTM_Predicted_Close"])
    baseline_metrics = calculate_metrics(
        result_df["Actual_Close"], result_df["RandomForest_Predicted_Close"]
    )

    metrics = {
        "data_period": {"start": args.start, "end_exclusive": args.end},
        "lookback_days": args.lookback,
        "features": FEATURE_COLUMNS,
        "lstm": lstm_metrics,
        "random_forest_baseline": baseline_metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_plots(
        df,
        train_df,
        val_df,
        test_df,
        history,
        result_df,
        lstm_metrics,
        baseline_metrics,
    )
    write_visual_summary(lstm_metrics, baseline_metrics)
    write_report(
        df,
        train_df,
        val_df,
        test_df,
        lstm_metrics,
        baseline_metrics,
        len(history.history["loss"]),
    )

    model.save(OUTPUT_DIR / "nvda_lstm_model.keras")

    print("\nAssessment artifacts created:")
    print(f"- {DATA_DIR / 'NVDA_2021_to_2026_april_raw.csv'}")
    print(f"- {DATA_DIR / 'NVDA_2021_to_2026_april_cleaned.csv'}")
    print(f"- {OUTPUT_DIR / 'predictions.csv'}")
    print(f"- {OUTPUT_DIR / 'metrics.json'}")
    print(f"- {OUTPUT_DIR / '01_price_history.png'}")
    print(f"- {OUTPUT_DIR / '02_lstm_training_loss.png'}")
    print(f"- {OUTPUT_DIR / '03_actual_vs_predicted.png'}")
    print(f"- {OUTPUT_DIR / '04_lstm_residuals.png'}")
    print(f"- {OUTPUT_DIR / '05_train_validation_test_split.png'}")
    print(f"- {OUTPUT_DIR / '06_feature_correlation_heatmap.png'}")
    print(f"- {OUTPUT_DIR / '07_model_error_comparison.png'}")
    print(f"- {OUTPUT_DIR / '08_lstm_absolute_error_over_time.png'}")
    print(f"- {OUTPUT_DIR / '09_daily_return_distribution.png'}")
    print(f"- {OUTPUT_DIR / '10_workflow_diagram.png'}")
    print("- visual_summary.html")
    print("- report.md")
    print("\nLSTM metrics:", json.dumps(lstm_metrics, indent=2))
    print("Random Forest metrics:", json.dumps(baseline_metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVDA LSTM stock-price prediction assessment.")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--lookback", type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
