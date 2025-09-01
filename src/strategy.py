import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from pykalman import KalmanFilter

def apply_strategy_donchian_breakout(df: pd.DataFrame) -> pd.DataFrame:
    """Donchian Channel Breakout strategy:
    - Buy if Close > Donchian_High
    - Sell if Close < Donchian_Low
    - Hold otherwise
    """
    df = df.copy()
    df["Signal"] = 0
    df.loc[df["Close"] > df["Donchian_High"], "Signal"] = 1
    df.loc[df["Close"] < df["Donchian_Low"], "Signal"] = -1
    return df

def apply_strategy_donchian_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """Donchian + RSI Filter strategy:
    - Buy if Close > Donchian_High AND RSI < 70
    - Sell if Close < Donchian_Low AND RSI > 30
    """
    df = df.copy()
    df["Signal"] = 0
    buy_condition = (df["Close"] > df["Donchian_High"]) & (df["RSI"] < 70)
    sell_condition = (df["Close"] < df["Donchian_Low"]) & (df["RSI"] > 30)
    df.loc[buy_condition, "Signal"] = 1
    df.loc[sell_condition, "Signal"] = -1
    return df

def apply_strategy_hmm(df: pd.DataFrame, n_states=3) -> pd.DataFrame:
    """Improved HMM strategy:
    - Uses daily pct_change instead of absolute Close price
    - Generates signal only on change of hidden state
    """
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df = df.dropna(subset=["Returns"])
    
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=1000)
    df_close = df["Returns"].values.reshape(-1, 1)
    model.fit(df_close)
    
    states = model.predict(df_close)
    df["HMM_State"] = states

    # Generate signals only on state change
    df["Signal"] = 0
    df["Prev_State"] = df["HMM_State"].shift(1)
    df.loc[df["HMM_State"] > df["Prev_State"], "Signal"] = 1   # upward transition
    df.loc[df["HMM_State"] < df["Prev_State"], "Signal"] = -1  # downward transition
    df = df.drop(columns=["Prev_State"])
    
    return df

def apply_strategy_kalman(df: pd.DataFrame) -> pd.DataFrame:
    """Kalman Filter strategy:
    - Smooth Close price with Kalman filter
    - Buy if price > smoothed trend, Sell if price < smoothed trend
    """
    df = df.copy()
    kf = KalmanFilter(initial_state_mean=df["Close"].values[0], n_dim_obs=1)
    state_means, _ = kf.filter(df["Close"].values)
    df["Kalman_Price"] = state_means

    df["Signal"] = 0
    df.loc[df["Close"] > df["Kalman_Price"], "Signal"] = 1
    df.loc[df["Close"] < df["Kalman_Price"], "Signal"] = -1
    return df
