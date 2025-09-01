import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features import add_indicators
from strategy import (
    apply_strategy_donchian_breakout,
    apply_strategy_donchian_rsi,
    apply_strategy_hmm,
    apply_strategy_kalman,
)
from visualization import plot_signals


def calculate_trade_returns(df: pd.DataFrame) -> list:
    """Calculate returns of each closed trade."""
    trade_returns = []
    position = 0
    entry_price = 0
    for i, row in df.iterrows():
        if row["Signal"] == 1 and position <= 0:
            entry_price = row["Close"]
            position = 1
        elif row["Signal"] == -1 and position >= 0:
            if position == 1:
                trade_returns.append((row["Close"] - entry_price) / entry_price)
            entry_price = row["Close"]
            position = -1
    return trade_returns


def save_table_as_image(df, filename="results_table.png", max_fontsize=12, min_fontsize=6):
    """Save DataFrame as PNG table with auto-scaled font to fit text."""
    nrows, ncols = df.shape
    fig, ax = plt.subplots(figsize=(ncols * 2, (nrows + 1) * 0.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )

    max_text_len = max(
        [len(str(cell)) for row in df.values for cell in row] +
        [len(str(col)) for col in df.columns]
    )

    fontsize = max(min(max_fontsize, 200 / max_text_len), min_fontsize)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    for key, cell in table.get_celld().items():
        cell.set_height(0.5)
        cell.set_width(1.0 / ncols)

    plt.savefig(filename, bbox_inches="tight", dpi=200)
    plt.close()


def backtest():
    # Load historical TSLA data
    data = yf.download("TSLA", start="2020-01-01", end=None)
    data = add_indicators(data, lookback=20, rsi_period=14)

    # Define strategies
    strategies = {
        "Donchian Breakout": apply_strategy_donchian_breakout,
        "Donchian + RSI": apply_strategy_donchian_rsi,
        "HMM Strategy": apply_strategy_hmm,
        "Kalman Filter": apply_strategy_kalman
    }

    results = []
    strategy_dfs = {}  # store DataFrames for each strategy

    for name, func in strategies.items():
        df = func(data.copy())

        # Calculate daily returns according to strategy
        df["Returns"] = df["Close"].pct_change() * df["Signal"].shift(1)
        df["Equity"] = (1 + df["Returns"].fillna(0)).cumprod()
        df["BuyHold"] = (1 + df["Close"].pct_change().fillna(0)).cumprod()

        strategy_dfs[name] = df

        # Metrics
        total_return = df["Equity"].iloc[-1] - 1
        sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252) if df["Returns"].std() != 0 else np.nan
        sortino = (df["Returns"].mean() / df[df["Returns"] < 0]["Returns"].std()) * np.sqrt(252) if not df[df["Returns"] < 0]["Returns"].empty else np.nan
        win_rate = (df["Returns"] > 0).sum() / max((df["Returns"] != 0).sum(), 1) * 100

        sum_gain = df.loc[df["Returns"] > 0, "Returns"].sum()
        sum_loss = abs(df.loc[df["Returns"] < 0, "Returns"].sum())
        profit_factor = sum_gain / sum_loss if sum_loss != 0 else np.nan

        trade_returns = calculate_trade_returns(df)
        average_trade_return = np.mean(trade_returns) if trade_returns else np.nan

        results.append({
            "Strategy": name,
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Win Rate": win_rate,
            "Profit Factor": profit_factor,
            "Average Trade Return": average_trade_return
        })

        # Plot signals
        plot_signals(df, strategy_name=name)

        # Plot equity curve vs buy-and-hold
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df["Equity"], label=f"{name} Equity")
        plt.plot(df.index, df["BuyHold"], label="Buy & Hold TSLA", linestyle="--")
        plt.title(f"{name} - Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity / Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.show()

    results_df = pd.DataFrame(results)

    # Save table as PNG
    save_table_as_image(results_df, "results_table.png")

    # Print table in console
    print("\nStrategy Performance (all strategies):")
    print(results_df)

    # Best strategy by total return
    best_strategy_name = results_df.loc[results_df["Total Return"].idxmax(), "Strategy"]
    best_df = strategy_dfs[best_strategy_name]
    best_total_return = results_df["Total Return"].max()
    print(f"\nBest strategy among 4: {best_strategy_name} ({best_total_return:.2%})")

    # Compare with Buy & Hold
    buyhold_return = (1 + data["Close"].pct_change().fillna(0)).cumprod().iloc[-1] - 1
    print(f"Buy & Hold Total Return: {buyhold_return:.2%}")

    if buyhold_return > best_total_return:
        print("Buy & Hold was more profitable than the best strategy.")
        return None, data, buyhold_return
    else:
        print(f"{best_strategy_name} outperformed Buy & Hold.")
        return best_strategy_name, best_df, buyhold_return


if __name__ == "__main__":
    backtest()
