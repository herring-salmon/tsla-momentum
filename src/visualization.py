import matplotlib.pyplot as plt

def plot_signals(df, strategy_name="Strategy"):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue")

    # Buy signals
    plt.plot(df[df["Signal"] == 1].index,
             df[df["Signal"] == 1]["Close"],
             "^", markersize=10, color="green", label="Buy Signal")

    # Sell signals
    plt.plot(df[df["Signal"] == -1].index,
             df[df["Signal"] == -1]["Close"],
             "v", markersize=10, color="red", label="Sell Signal")

    plt.title(f"{strategy_name} - Buy/Sell Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
