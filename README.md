# market_forecast
Python codebase for predicting future stock market trends.

Built a Geometric Brownian Motion (GBM)/Long Short-Term Memory (LSTM)/Reinforcement Learning (RL) fused model for long-term prediction. Adjustable initialization period for data training. 

Additional GBM/LSTM model fused with VADER for sentiment analysis of weekly news articles pertaining to stock of interest. This model is for short term prediction. 

Lastly includes a model for predicting profit on options (long call and long put) using real time Yahoo Finance data and Black Scholes (Greeks).

Both stock pricing predictive models provide graphics of the historical data (actual + predicted), alongside the trend of the predicted future prices. The options model also provides a graph of the profit versus the intended long call or long put choice.
