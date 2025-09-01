import matplotlib.pyplot as plt
from backtest import backtest
from visualization import plot_signals

def main():
    # Run the backtest
    best_strategy_name, best_df, buyhold_return = backtest()

    if best_strategy_name is None:
        # Buy & Hold was more profitable
        print(f"\nBuy & Hold TSLA was more profitable: {buyhold_return:.2%}")
        plt.figure(figsize=(14, 7))
        plt.plot(best_df.index, (1 + best_df["Close"].pct_change().fillna(0)).cumprod(), 
                 label="Buy & Hold TSLA", linestyle="--")
        plt.title("Buy & Hold TSLA")
        plt.xlabel("Date")
        plt.ylabel("Equity / Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # Best strategy among the 4
        print(f"\nBest strategy: {best_strategy_name}")
        # Plot buy/sell signals
        plot_signals(best_df, strategy_name=best_strategy_name)
        # Plot equity curve vs Buy & Hold
        plt.figure(figsize=(14, 7))
        plt.plot(best_df.index, best_df["Equity"], label=f"{best_strategy_name} Equity")
        plt.plot(best_df.index, (1 + best_df["Close"].pct_change().fillna(0)).cumprod(), 
                 label="Buy & Hold TSLA", linestyle="--")
        plt.title(f"{best_strategy_name} - Equity Curve vs Buy & Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity / Cumulative Returns")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
