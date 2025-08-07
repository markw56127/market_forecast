import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
import ta
import random
import gymnasium as gym
from gymnasium import spaces

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, dropout=0.3, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
    
class AllocationEnv(gym.Env):
    def __init__(self, prices, indicators, horizon=30):
        super().__init__()
        self.prices = prices
        self.indicators = indicators
        self.horizon = horizon
        self.actions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(indicators.shape[1],), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self.indicators[self.idx], {}

    def step(self, action_idx):
        alloc = self.actions[action_idx]
        if self.idx + self.horizon >= len(self.prices):
            reward = 0.0
            terminated = True
        else:
            initial = self.prices[self.idx]
            final = self.prices[self.idx + self.horizon]
            pct_return = (final - initial) / initial
            reward = alloc * pct_return
            self.idx += 1
            terminated = False

        truncated = False
        obs = self.indicators[self.idx] if self.idx < len(self.prices) else self.indicators[-1]
        return obs, reward, terminated, truncated, {}

def get_nearest_date(dates_index, target_str, direction='forward'):
    target = pd.to_datetime(target_str)
    if target in dates_index:
        return target
    valid = dates_index[dates_index >= target] if direction == 'forward' else dates_index[dates_index <= target]
    if len(valid) == 0:
        raise ValueError(f"No valid date found near {target_str}")
    return valid[0] if direction == 'forward' else valid[-1]

def prepare_sequences(multivariate_series, seq_len):
    X, y = [], []
    for i in range(seq_len, len(multivariate_series)):
        X.append(multivariate_series[i - seq_len:i])
        y.append(multivariate_series[i, 0])  # predict closing price
    return np.array(X), np.array(y)

def plot_predictions(dates, actual, predicted, future_dates=None, future_preds=None, ticker=str):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    if future_dates is not None and future_preds is not None:
        plt.plot(future_dates, future_preds, label="Future Prediction", linestyle='--')
    plt.title(f"LSTM + GBM Hybrid Forecast: " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def simulate_gbm(S0, mu, sigma, T=1, N=252):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) * np.sqrt(dt)
    W = np.cumsum(W)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

def run_lstm_forecast(
    ticker: str,
    train_period: tuple,
    test_period: tuple,
    future_days: int = 0,
    seq_len: int = 60,
    batch_size: int = 32,
    epochs: int = 30,
    hidden_size: int = 100,
    num_layers: int = 3,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    gbm_weight: float = 0.3
):
    print(f"Downloading {ticker} data from {train_period[0]} to {test_period[1]}...")
    df = yf.download(ticker, start=train_period[0], end=test_period[1])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if df.empty:
        raise ValueError("No data downloaded. Check ticker and date range.")
    df.ffill(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    close_series = df['Close'].squeeze()

    df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    df['SMA_20'] = ta.trend.SMAIndicator(close=close_series, window=20).sma_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(close=close_series, window=20).ema_indicator()
    df['MACD'] = ta.trend.MACD(close=close_series).macd()
    df.dropna(inplace=True)

    close_prices = df['Close']

    train_start = get_nearest_date(df.index, train_period[0], 'forward')
    train_end = get_nearest_date(df.index, train_period[1], 'backward')
    test_start = get_nearest_date(df.index, test_period[0], 'forward')
    test_end = get_nearest_date(df.index, test_period[1], 'backward')

    print(f"Adjusted Train: {train_start.date()} → {train_end.date()}")
    print(f"Adjusted Test: {test_start.date()} → {test_end.date()}")

    train_df = df.loc[train_start:train_end]
    test_df = df.loc[test_start:test_end]

    features = ['Close', 'RSI', 'SMA_20', 'EMA_20', 'MACD']
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    X_train, y_train = prepare_sequences(train_scaled, seq_len)
    X_test, y_test = prepare_sequences(test_scaled, seq_len)

    test_dates = test_df.index[seq_len:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE).view(-1, 1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[2]
    model = DeepLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Training model...")
    model.train()
    for epoch in range(epochs):
        loss_total = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_total/len(train_dl):.6f}")

    print("Evaluating on test data...")
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    preds_rescaled = close_scaler.inverse_transform(preds)
    y_rescaled = close_scaler.inverse_transform(y_test_t.cpu().numpy())

    future_preds_rescaled = None
    future_dates = None

    if future_days > 0:
        print(f"Predicting {future_days} days into the future with LSTM...")
        model_input = X_test_t[-1:].clone()
        future_preds_scaled = []

        for _ in range(future_days):
            with torch.no_grad():
                pred_scaled = model(model_input).cpu().numpy()
            future_preds_scaled.append(pred_scaled[0, 0])
            model_input_np = model_input.cpu().numpy()
            next_features = model_input_np[:, -1, :].copy()
            next_features[0, 0] = pred_scaled

            next_input_np = np.concatenate((model_input_np[:, 1:, :], next_features.reshape(1, 1, -1)), axis=1)
            model_input = torch.tensor(next_input_np, dtype=torch.float32).to(DEVICE)

        future_preds_scaled = np.array(future_preds_scaled).reshape(-1, 1)
        future_preds_rescaled = close_scaler.inverse_transform(future_preds_scaled).flatten()

        print("Simulating GBM for future...")
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        mu = float(log_returns.mean() * 252)
        sigma = float(log_returns.std() * np.sqrt(252))
        S0 = float(close_prices.iloc[-1])
        gbm_sim = simulate_gbm(S0, mu, sigma, T=future_days / 252, N=future_days)
        hybrid_future = gbm_weight * gbm_sim + (1 - gbm_weight) * future_preds_rescaled
        last_date = df.index[-1]
        future_dates = pd.bdate_range(last_date, periods=future_days + 1)[1:]
        future_preds_rescaled = hybrid_future

    plot_predictions(test_dates, y_rescaled.flatten(), preds_rescaled.flatten(), future_dates, future_preds_rescaled, ticker)

    if future_preds_rescaled is not None:
        print("Starting RL optimization for future decision-making...")
        
        indicators = test_scaled[-len(future_preds_rescaled):, 1:]
        env = AllocationEnv(np.array(future_preds_rescaled), indicators, horizon=30)

        total_reward = 0
        obs, _ = env.reset()
        for _ in range(len(future_preds_rescaled) - 30):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break

        print(f"Simulated reward over horizon: {total_reward:.2f}")


if __name__ == "__main__":
    run_lstm_forecast(
        ticker="IBIT",
        train_period=("2013-01-01", "2025-001-01"),
        test_period=("2025-01-01", "2025-08-06"),
        future_days=60,
        epochs=50,
        gbm_weight=0.3
    )
