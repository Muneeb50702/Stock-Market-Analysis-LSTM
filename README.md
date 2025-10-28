# Stock Market Analysis with LSTM

Predict next-day High, Low, and Close prices for selected US tickers using an LSTM model trained on technical indicators, cyclical time features, and one‑hot encoded tickers.

## Overview

This project builds and serves a sequence model (LSTM) that consumes the last 10 trading days of engineered features and predicts the next day’s targets: `High`, `Low`, and `Close`.

- Targets: 3 continuous variables — High, Low, Close
- Sequence length: 10 timesteps
- Feature size: 26 features per timestep
- Supported tickers: AAPL, AMZN, BRK-B, GOOGL, JPM, META, MSFT, NVDA, TSLA, V
- Inference script: `test.py` (downloads data via yfinance, computes indicators with TA‑Lib, scales inputs, runs the trained model, and prints predictions vs actuals)

Mathematically, the model input and output during inference follow:

- Input: $X \in \mathbb{R}^{1 \times 10 \times 26}$ (batch, timesteps, features)
- Output: $\hat{y} \in \mathbb{R}^{1 \times 3}$ (High, Low, Close)

## Repository structure

- `data_preprocessing.ipynb` — Build features (technical indicators, cyclical time features, one‑hot tickers) from raw data
- `splitting_and_training.ipynb` — Train/validate an LSTM, scale features/targets, and export artifacts
- `TEST.ipynb` — Scratch/testing notebook (exploration/plots)
- `test.py` — CLI inference using yfinance + TA‑Lib + saved `model.keras` and scalers
- `cleaned_stock_data.csv` — Prepared dataset used in notebooks
- `to_be_done.csv` — Additional data (backlog/todo)
- `model.keras` — Saved Keras model (TensorFlow 2.x format)
- `x_scaler_minmax.pkl` — Fitted `MinMaxScaler` for features (X)
- `y_scaler_minmax.pkl` — Fitted `MinMaxScaler` for targets (y)
- `features.json` — List of feature column names (26)
- `targets.json` — List of target column names (3)

## Engineered features

The 26 feature columns used by both training and inference (see `features.json`):

1. Price/volume: `Open`, `Volume`
2. Moving averages: `SMA_10`, `SMA_50`, `EMA_20`
3. Trend/momentum: `MACD`, `MACD_signal`, `RSI`
4. Volatility/bands: `BB_upper`, `BB_lower`
5. Stochastics: `slowk`, `slowd`
6. One‑hot ticker flags: `Ticker_AAPL`, `Ticker_AMZN`, `Ticker_BRK-B`, `Ticker_GOOGL`, `Ticker_JPM`, `Ticker_META`, `Ticker_MSFT`, `Ticker_NVDA`, `Ticker_TSLA`, `Ticker_V`
7. Cyclical time features: `Month_sin`, `Month_cos`, `Day_sin`, `Day_cos`

Notes:
- `MACD_hist`, `BB_middle`, raw `Day`/`Month`, and non‑price corporate actions (`Dividends`, `Stock Splits`) are dropped during preprocessing.
- Time features use cyclical encoding to preserve periodicity: e.g., `Month_sin = sin(2π·month/12)`.

Targets (from `targets.json`): `High`, `Low`, `Close`.

## How it works

1. Data sourcing (in `test.py`):
   - Downloads ~80 days of daily bars via yfinance `Ticker.history(period="80d", interval="1d")`.
   - Computes indicators with TA‑Lib and builds the 26 features.
   - One‑hot encodes the chosen ticker.
2. Windowing:
   - Sorts data by date, drops NAs, keeps the most recent 10 timesteps → shape `(1, 10, 26)`.
3. Scaling:
   - Loads fitted `MinMaxScaler`s from `x_scaler_minmax.pkl` (X) and `y_scaler_minmax.pkl` (y) and applies them consistently.
4. Inference:
   - Loads `model.keras`, predicts the next day `[High, Low, Close]`, then inverse‑transforms to original price scale.

Training is handled in the notebooks:
- `data_preprocessing.ipynb` prepares features and targets from `cleaned_stock_data.csv`.
- `splitting_and_training.ipynb` performs train/validation split, scales data, defines/trains the LSTM, and saves `model.keras` + scalers.

## Getting started

### Prerequisites

- Python 3.9 or newer (3.10 recommended)
- macOS users: Xcode CLT (for some wheels), Homebrew is helpful

### Installation (virtual environment + packages)

```bash
# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip
python -m pip install --upgrade pip

# core dependencies
pip install tensorflow==2.15.0 \
            numpy pandas scikit-learn yfinance \
            matplotlib seaborn plotly

# TA‑Lib (technical indicators)
# On macOS, first install the native library:
#   brew install ta-lib
# Then install the Python wrapper:
pip install TA-Lib
```

If `brew` is unavailable or you hit build errors for TA‑Lib, search for a prebuilt wheel matching your Python version and platform, or use a conda environment:

```bash
# optional alternative via conda
conda install -c conda-forge ta-lib
```

### Quickstart: run inference

```bash
python test.py
```

When prompted, enter a ticker from:

```
AAPL, AMZN, BRK-B, GOOGL, JPM, META, MSFT, NVDA, TSLA, V
```

The script will download the latest data, compute indicators, and print:

```
Predicted: [<High_pred> <Low_pred> <Close_pred>]
Actual:    [<High_actual> <Low_actual> <Close_actual>]
```

Requirements for inference:
- `model.keras`, `x_scaler_minmax.pkl`, `y_scaler_minmax.pkl` present in the repo root.
- Working internet connection (for yfinance).
- TA‑Lib installed correctly.

### Train (optional, via notebooks)

1. Open `data_preprocessing.ipynb` and run cells top‑to‑bottom to build the feature set from `cleaned_stock_data.csv`.
2. Open `splitting_and_training.ipynb` and run the training pipeline. It should export:
   - `model.keras`
   - `x_scaler_minmax.pkl`
   - `y_scaler_minmax.pkl`

Tip: The feature/target columns are aligned with `features.json` and `targets.json`. If you alter features, update both the notebooks and `test.py` accordingly.

## Configuration

- Edit `features.json` and `targets.json` to define the model’s input and output columns. Keep them synchronized with both the notebooks and `test.py`.
- Sequence length is currently 10; changing it requires updating the training pipeline and the inference reshaping in `test.py`.
- The supported ticker list is hard‑coded in `test.py` via one‑hot columns; to add tickers, extend the one‑hot set consistently across data prep, training, and inference.

## Tips, caveats, and troubleshooting

- TA‑Lib install issues (macOS):
  - `brew install ta-lib` before `pip install TA-Lib`.
  - If you still see build errors, try conda (`conda install -c conda-forge ta-lib`) or prebuilt wheels.
- TensorFlow on macOS:
  - CPU works out‑of‑the‑box. For Apple Silicon acceleration, consider `tensorflow-macos`/`tensorflow-metal` (adjust versions accordingly).
- yfinance rate limits or network issues:
  - Retry after a pause; ensure the ticker symbol matches the allowed set.
- Shape or scaler mismatches:
  - Confirm `features.json` has exactly 26 features matching the order used in training and `test.py`.
  - Ensure you’re feeding `(1, 10, 26)` into the model after scaling with the same `x_scaler_minmax.pkl`.
- Missing `model.keras`/scalers:
  - Re‑run the training notebooks to regenerate artifacts, or copy the files into the repo root.

## Extending the project

- Add CLI args to `test.py` (e.g., `--ticker`, `--period`, `--window`), and optional CSV output.
- Log metrics and add backtesting plots (e.g., rolling MAPE/RMSE).
- Support more tickers or a generic ticker embedding instead of one‑hot.
- Export a `requirements.txt` or `pyproject.toml` for reproducible environments.
- Package the pipeline as a module with unit tests and CI.

## License

No license file is provided in this repository. If you intend to use or distribute this code, consider adding a license.

---

If you have specific requirements (deployment, additional tickers, or alternative features), open an issue or adapt the notebooks and `test.py` following the guidance above.
