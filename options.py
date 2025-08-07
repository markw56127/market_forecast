import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from xgboost import XGBRegressor
from ta.momentum import RSIIndicator
from ta.trend import MACD

def download_stock_data(ticker, period='1y'):
    df = yf.download(ticker, period=period)
    df = df[['Close']]
    return df

def predict_next_price(ticker):
    df = yf.download(ticker, period='6mo', interval='1d')
    df_close = df['Close']

    df['Return'] = df_close.pct_change()
    df_close_1d = pd.Series(df_close.values.flatten(), index=df_close.index)
    df['RSI'] = RSIIndicator(df_close_1d, window=14).rsi()
    macd = MACD(df_close_1d)
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()

    df.dropna(inplace=True)

    df['TargetReturn'] = df['Return'].shift(-1)
    df.dropna(inplace=True)

    X = df[['Close', 'Return', 'RSI', 'MACD', 'Signal']]
    y = df['TargetReturn']

    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X, y)

    last_row = X.iloc[-1:]
    pred_return = model.predict(last_row)[0]
    last_close_price = float(df_close.iloc[-1])
    pred_price = last_close_price * (1 + pred_return)
    pred_price = float(pred_price)

    print("Predicted return from model:", pred_return)
    print("Predicted price from return:", pred_price)

    return pred_price

def get_option_chain(ticker, expiry=None):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if expiry is None:
        expiry = expirations[0] 
    option_chain = stock.option_chain(expiry)
    calls = option_chain.calls
    puts = option_chain.puts
    return calls, puts, expiry

def option_payoff(price_range, strike, option_type='call', premium=0):
    if option_type == 'call':
        payoff = np.maximum(price_range - strike, 0) - premium
    elif option_type == 'put':
        payoff = np.maximum(strike - price_range, 0) - premium
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return payoff

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2*np.sqrt(T))) + r * K * np.exp(-r*T) * norm.cdf(-d2)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return float(delta), float(gamma), float(theta), float(vega)

def plot_option_payoffs(price_range, payoffs, labels, title, break_evens=None):
    plt.figure(figsize=(10,6))
    for payoff, label in zip(payoffs, labels):
        plt.plot(price_range, payoff, label=label)

    if break_evens:
        for price, label in break_evens:
            plt.axvline(price, color='gray', linestyle='--', alpha=0.5, label=label)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.title(title)
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_strategies(ticker):
    print(f"Downloading option chain for {ticker}...")
    calls, puts, expiry = get_option_chain(ticker)
    print(f"Options expiry date: {expiry}")

    print("\nTop 5 Calls by Volume:")
    print(calls.sort_values('volume', ascending=False)[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest']].head())

    print("\nTop 5 Puts by Volume:")
    print(puts.sort_values('volume', ascending=False)[['contractSymbol','strike','lastPrice','bid','ask','volume','openInterest']].head())

    stock_price = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
    print(f"\nCurrent stock price: {stock_price:.2f}")

    # Use ATM options
    atm_call = calls.iloc[(calls['strike'] - stock_price).abs().argsort()[0]]
    atm_put = puts.iloc[(puts['strike'] - stock_price).abs().argsort()[0]]

    call_strike = atm_call['strike']
    call_premium = atm_call['lastPrice']
    put_strike = atm_put['strike']
    put_premium = atm_put['lastPrice']

    price_range = np.linspace(stock_price*0.5, stock_price*1.5, 200)

    call_payoff = option_payoff(price_range, call_strike, 'call', premium=call_premium)
    put_payoff = option_payoff(price_range, put_strike, 'put', premium=put_premium)

    short_call_payoff = -option_payoff(price_range, call_strike, 'call', premium=call_premium)
    covered_call_payoff = (price_range - stock_price) + short_call_payoff

    predicted_price = predict_next_price(ticker)
    print(f"Predicted next-day price: {predicted_price:.2f}")

    print(f"\nExpected payoff at predicted price:")
    print(f"Long Call: {option_payoff(np.array([predicted_price]), call_strike, 'call', call_premium)[0]:.2f}")
    print(f"Long Put: {option_payoff(np.array([predicted_price]), put_strike, 'put', put_premium)[0]:.2f}")
    print(f"Covered Call: {covered_call_payoff[np.abs(price_range - predicted_price).argmin()]:.2f}")

    T_days = (pd.to_datetime(expiry) - pd.Timestamp.today()).days
    T = T_days / 365
    r = 0.05
    sigma = np.log(download_stock_data(ticker)['Close']).diff().dropna().std() * np.sqrt(252)

    print(f"\nOption Greeks (using BSM):")
    call_greeks = black_scholes_greeks(stock_price, call_strike, T, r, sigma, 'call')
    put_greeks = black_scholes_greeks(stock_price, put_strike, T, r, sigma, 'put')

    print(f"Call (Strike {call_strike}): Delta={call_greeks[0]:.3f}, Gamma={call_greeks[1]:.3f}, Theta={call_greeks[2]:.2f}, Vega={call_greeks[3]:.2f}")
    print(f"Put  (Strike {put_strike}): Delta={put_greeks[0]:.3f}, Gamma={put_greeks[1]:.3f}, Theta={put_greeks[2]:.2f}, Vega={put_greeks[3]:.2f}")

    print(f"\nP/L at current price ({stock_price:.2f}):")
    print(f"Long Call: {option_payoff(np.array([stock_price]), call_strike, 'call', call_premium)[0]:.2f}")
    print(f"Long Put: {option_payoff(np.array([stock_price]), put_strike, 'put', put_premium)[0]:.2f}")
    print(f"Covered Call: {covered_call_payoff[np.abs(price_range - stock_price).argmin()]:.2f}")

    plot_option_payoffs(
        price_range,
        [call_payoff, put_payoff, covered_call_payoff],
        [f'Long Call (Strike {call_strike})',
         f'Long Put (Strike {put_strike})',
         f'Covered Call (Short {call_strike})'],
        f'{ticker} Option Strategy Payoffs (Exp: {expiry})',
        break_evens=[
            (call_strike + call_premium, 'Call B/E'),
            (put_strike - put_premium, 'Put B/E')
        ]
    )

if __name__ == "__main__":
    simulate_strategies("AAPL")
