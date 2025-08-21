import yfinance as yf
import talib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import json
from sklearn.preprocessing import MinMaxScaler
import pickle

current_ticker = input("Enter the ticker from (AAPL, AMZN, BRK-B, GOOGL, JPM, META, MSFT, NVDA, TSLA, V): ").strip().upper()
stock = yf.Ticker(current_ticker)
df = stock.history(period="80d", interval="1d")
df = df.reset_index()
feature_cols = ['Open', 'Volume', 'SMA_10', 'SMA_50', 'EMA_20',
               'MACD', 'MACD_signal', 'RSI', 'BB_upper',
               'BB_lower', 'slowk', 'slowd', 'Ticker_AAPL', 'Ticker_AMZN',
               'Ticker_BRK-B', 'Ticker_GOOGL', 'Ticker_JPM', 'Ticker_META',
               'Ticker_MSFT', 'Ticker_NVDA', 'Ticker_TSLA', 'Ticker_V',
               'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']

target_cols = ['High', 'Low', 'Close']

ticker_cols = ['Ticker_AAPL', 'Ticker_AMZN', 'Ticker_BRK-B', 'Ticker_GOOGL', 
                'Ticker_JPM', 'Ticker_META', 'Ticker_MSFT', 'Ticker_NVDA', 
                'Ticker_TSLA', 'Ticker_V']


df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'], timeperiod=20)
df['slowk'], df['slowd'] = talib.STOCH(df['High'], df['Low'], df['Close'])
for ticker in ticker_cols:
    df[ticker] = 1 if ticker == f'Ticker_{current_ticker}' else 0

df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
df = df.drop(columns=['Dividends','Stock Splits', 'MACD_hist', 'BB_middle', 'Day', 'Month'])
df = df.dropna()
df = df.sort_values(by='Date').reset_index(drop=True)

df = df.drop(['Date'], axis = 1)
target = df[target_cols].copy()
df.drop(columns=target_cols, inplace=True)
df = df.drop(df.index[-1])
last_10_timesteps = df[-10:]  # Shape will be (10, 26)
X_input = np.reshape(last_10_timesteps, (1, 10, 26))  # Shape: (1, 10, 26)



with open('x_scaler_minmax.pkl', 'rb') as f:
    x_scaler = pickle.load(f)

with open('y_scaler_minmax.pkl', 'rb') as f:
    y_scaler = pickle.load(f)

original_shape = X_input.shape

inputX_2d = X_input.reshape(-1, original_shape[2])  

inputX_scaled_2d = x_scaler.transform(inputX_2d)

inputX_scaled = inputX_scaled_2d.reshape(original_shape)
model = load_model('model.keras')
predictions = model.predict(inputX_scaled)
predictions = y_scaler.inverse_transform(predictions)
actual = target[-1:]
actual = np.array(actual)
print(f"Predicted: {predictions.flatten()}")
print(f"Actual: {actual.flatten()}")