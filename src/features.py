import pandas as pd
import ta

def add_indicators(df, lookback=20, rsi_period=14):
    """
    Add technical indicators to the stock DataFrame.
    """
    df = df.copy()
    
    # Убираем MultiIndex, если есть
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Преобразуем в Series
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    
    # Donchian channel
    df['Donchian_High'] = high.rolling(window=lookback).max().shift(1)
    df['Donchian_Low'] = low.rolling(window=lookback).min().shift(1)
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=close, window=rsi_period).rsi()
    
    # Убираем первые NaN значения
    df = df.dropna(subset=['Donchian_High', 'Donchian_Low', 'RSI'])
    
    return df
