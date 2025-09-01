# TSLA Momentum Trading Model

This project implements multiple **momentum trading strategies** for Tesla stock (TSLA) and evaluates them on historical data. It also generates buy/sell signals in near real-time using Yahoo Finance data.

## Project Overview

- **Data source**: Yahoo Finance (`yfinance` library)
- **Strategies implemented**:
  1. **Donchian Breakout** – Buy when price breaks above the highest high over the last N days; sell when below lowest low.
  2. **Donchian + RSI Filter** – Combines Donchian breakout with RSI overbought/oversold conditions.
  3. **HMM Strategy** – Uses Hidden Markov Model to detect market regimes and generate signals on state transitions.
  4. **Kalman Filter Strategy** – Smooths price with Kalman filter and generates signals when price deviates from the trend.

- **Features**:  
  - Donchian channel highs/lows  
  - RSI (Relative Strength Index)  

- **Backtesting**:
  - Equity curve vs. buy-and-hold
  - Daily and trade-level returns
  - Metrics: Total Return, Sharpe Ratio, Sortino Ratio, Win Rate, Profit Factor, Average Trade Return

- **Visualization**:
  - Buy/sell signals on price chart
  - Equity curve comparison
  - Backtest results table saved as PNG

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tsla-momentum
