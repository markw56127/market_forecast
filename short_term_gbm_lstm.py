import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, TensorDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from yahoo_fin import news
import time

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

def add_technical_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    df = df.ffill().bfill()
    return df

def get_nearest_date(dates_index, target_str, direction='forward'):
    target = pd.to_datetime(target_str)
    if target in dates_index:
        return target
    valid = dates_index[dates_index >= target] if direction == 'forward' else dates_index[dates_index <= target]
    if len(valid) == 0:
        raise ValueError(f"No valid date found near {target_str}")
    return valid[0] if direction == 'forward' else valid[-1]

def prepare_sequences(scaled_features, seq_len):
    X, y = [], []
    for i in range(seq_len, len(scaled_features)):
        X.append(scaled_features[i - seq_len:i])
        y.append(scaled_features[i, 0])  # Close price target
    return np.array(X), np.array(y)

def simulate_gbm(S0, mu, sigma, T=1, N=252):
    dt = T / N
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N) * np.sqrt(dt)
    W = np.cumsum(W)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

def fetch_recent_yahoo_news_sentiment(ticker, end_date, days=7, max_articles=5):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    dates = pd.bdate_range(end=end_date, periods=days)
    
    try:
        articles = news.get_yf_rss(ticker)
    except Exception as e:
        print(f"Error fetching news RSS: {e}")
        return pd.Series([0.0]*days, index=dates)

    # Group articles by date
    articles_by_date = {}
    for article in articles:
        ts = article.get('providerPublishTime')
        if ts is None:
            continue
        art_date = pd.to_datetime(ts, unit='s').normalize()
        if art_date not in articles_by_date:
            articles_by_date[art_date] = []
        if len(articles_by_date[art_date]) < max_articles:
            articles_by_date[art_date].append(article['title'])

    for dt in dates:
        headlines = articles_by_date.get(dt, [])
        if headlines:
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            sentiment_scores.append(np.mean(scores))
        else:
            sentiment_scores.append(0.0)
    return pd.Series(sentiment_scores, index=dates)

def plot_predictions(dates, actual, predicted, future_dates=None, future_preds=None, ticker=str):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    if future_dates is not None and future_preds is not None:
        plt.plot(future_dates, future_preds, label="Future Prediction", linestyle='--')
    plt.title("LSTM + GBM Hybrid Forecast: " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_lstm_forecast(
    ticker: str,
    train_period: tuple,
    test_period: tuple,
    future_days: int = 30,
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
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker and date range.")
    
    df = add_technical_indicators(df)

    print(f"Fetching recent Yahoo Finance news sentiment (last 7 days)...")
    sentiment_series = fetch_recent_yahoo_news_sentiment(ticker, df.index[-1], days=7, max_articles=5)

    # Initialize Sentiment column to 0 for all dates
    df['Sentiment'] = 0.0

    # Insert recent sentiment values for last 7 dates if they exist in df.index
    for date, score in sentiment_series.items():
        if date in df.index:
            df.at[date, 'Sentiment'] = score

    # Forward fill sentiment to propagate recent sentiment slightly into the test period
    df['Sentiment'] = df['Sentiment'].ffill().fillna(0)

    feature_cols = ['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'Momentum_10', 'RSI', 'Volatility_20', 'Sentiment']
    data = df[feature_cols].copy()
    data = data.ffill().bfill()

    train_start = get_nearest_date(df.index, train_period[0], 'forward')
    train_end = get_nearest_date(df.index, train_period[1], 'backward')
    test_start = get_nearest_date(df.index, test_period[0], 'forward')
    test_end = get_nearest_date(df.index, test_period[1], 'backward')

    print(f"Adjusted Train: {train_start.date()} → {train_end.date()}")
    print(f"Adjusted Test: {test_start.date()} → {test_end.date()}")

    train_data = data.loc[train_start:train_end]
    test_data = data.loc[test_start:test_end]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = prepare_sequences(train_scaled, seq_len)
    X_test, y_test = prepare_sequences(test_scaled, seq_len)
    test_dates = test_data.index[seq_len:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE).view(-1,1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE).view(-1,1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = DeepLSTMModel(input_size=len(feature_cols), hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(DEVICE)
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
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_total/len(train_dl):.6f}")

    print("Evaluating on test data...")
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
    preds_rescaled = scaler.inverse_transform(np.hstack((preds, np.zeros((preds.shape[0], len(feature_cols)-1)))))[:, 0]
    y_rescaled = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(feature_cols)-1)))))[:, 0]

    future_preds_rescaled = None
    future_dates = None

    if future_days > 0:
        print(f"Predicting {future_days} days into the future with LSTM...")

        model_input = X_test_t[-1:].clone()
        future_preds_scaled = []

        for day in range(future_days):
            with torch.no_grad():
                pred_scaled = model(model_input).cpu().numpy()  # shape (1,1)
            future_preds_scaled.append(pred_scaled[0, 0])

            model_input_np = model_input.cpu().numpy()

            # Prepare next timestep features:
            # Predicted Close price + last known other features (to keep feature dimension consistent)
            last_features = model_input_np[:, -1, 1:]  # shape (1, features-1)
            next_step_features = np.concatenate((pred_scaled.reshape(1, 1), last_features.reshape(1, -1)), axis=1)  # shape (1, features)
            next_step_features = next_step_features.reshape(1, 1, -1)  # (batch=1, seq_len=1, features)

            # Slide window: remove oldest step and append new step
            next_input_np = np.concatenate((model_input_np[:, 1:, :], next_step_features), axis=1)

            # Clamp values to [0,1] to match scaler range
            next_input_np = np.clip(next_input_np, 0, 1)

            # Convert to tensor for next iteration
            model_input = torch.tensor(next_input_np, dtype=torch.float32).to(DEVICE)

        future_preds_scaled = np.array(future_preds_scaled).reshape(-1,1)
        future_preds_rescaled = scaler.inverse_transform(np.hstack((future_preds_scaled, np.zeros((future_preds_scaled.shape[0], len(feature_cols)-1)))))[:,0]

        # GBM simulation
        print("Simulating GBM for future...")
        close_prices = df['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        mu = log_returns.mean() * 252
        sigma = log_returns.std() * np.sqrt(252)
        S0 = float(close_prices.iloc[-1])
        gbm_sim = simulate_gbm(S0, float(mu), float(sigma), T=future_days/252, N=future_days)

        hybrid_future = gbm_weight * gbm_sim + (1 - gbm_weight) * future_preds_rescaled

        last_date = df.index[-1]
        future_dates = pd.bdate_range(last_date, periods=future_days+1)[1:]
        future_preds_rescaled = hybrid_future

    plot_predictions(test_dates, y_rescaled, preds_rescaled, future_dates, future_preds_rescaled, ticker)

if __name__ == "__main__":
    run_lstm_forecast(
        ticker="AMD",
        train_period=("2013-01-01", "2025-08-07"),
        test_period=("2025-01-01", "2025-08-07"),
        future_days=30,
        epochs=50,
        gbm_weight=0.3,
        seq_len=60
    )
