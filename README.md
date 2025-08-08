# market_forecast
Python codebase for predicting future stock market trends.

This repository contains three Python scripts for:
- Geometric Brownian Motion (GBM)/Long Short-Term Memory (LSTM)/Reinforcement Learning (RL) fused model for long-term prediction. Adjustable initialization period for data training.
- High-dimensional LSTM + GBM forecasting with technical indicators and sentiment analysis from Yahoo Finance news.
- Model for option chain retrieval, payoff simulation, and Black–Scholes Greeks calculation.

Features include:
- Stock price forecasting with LSTM + GBM hybrid approach
- Reinforcement learning environment for allocation optimization
- Sentiment analysis integration
- Option payoff visualizations and Greeks

## Installation
```bash
git clone https://github.com/markw56127/market_forecast.git
cd market_forecast
pip install -r requirements.txt

python rl_gbm_lstm.py
python short_term_high_dimension_gbm_lstm.py
python options.py
