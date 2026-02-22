import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("🚀 AI Quant Forex Engine - Bug Proof")

pair = st.text_input("Forex Pair", "EURUSD=X")

capital = st.number_input("Capital (€)", value=1000)
risk_percent = st.slider("Risk % per trade", 0.5, 3.0, 1.0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if st.button("Run Analysis"):

    # Safe download
    data = yf.download(pair, period="1y", interval="1h", progress=False)

    if data is None or data.empty:
        st.error("Market data not available")
        st.stop()

    # Remove NaN early
    data = data.dropna()

    if len(data) < 250:
        st.warning("Not enough historical data")
        st.stop()

    # ===============================
    # Indicators
    # ===============================

    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()

    # RSI Stable Calculation
    delta = data["Close"].diff()

    gain = delta.clip(lower=0).ewm(span=14).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14).mean()

    loss = loss.replace(0, np.nan)

    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(50)

    # MACD Engine
    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9).mean()

    # ATR Volatility Engine
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = tr.rolling(14).mean()

    data = data.dropna()

    if data.empty:
        st.error("Indicator computation failed")
        st.stop()

    # Latest safe row extraction
    latest = data.iloc[-1]

    price = float(latest["Close"])
    ma50 = float(latest["MA50"])
    ma200 = float(latest["MA200"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_signal"])
    atr = float(latest["ATR"])

    # ===============================
    # Quant Scoring Engine
    # ===============================

    score = 0

    # Trend filter
    if ma50 > ma200:
        score += 2
    else:
        score -= 2

    # Momentum filter
    if macd > macd_signal:
        score += 1.5
    else:
        score -= 1.5

    # RSI neutral zone filter
    if 45 < rsi < 65:
        score += 1
    else:
        score -= 1

    # Volatility regime filter
    volatility_ratio = atr / price if price != 0 else 0

    if volatility_ratio < 0.01:
        score += 1
    else:
        score -= 1

    confidence = sigmoid(score)

    # ===============================
    # Signal Decision
    # ===============================

    signal = "HOLD"

    if confidence > 0.65:
        signal = "BUY"
    elif confidence < 0.35:
        signal = "SELL"

    # ===============================
    # Output
    # ===============================

    st.subheader("🧠 AI Output")

    st.write("Signal:", signal)
    st.write("Confidence:", round(confidence * 100, 2), "%")
    st.write("Price:", round(price, 5))

    # Risk Engine

    if signal != "HOLD" and atr > 0:

        risk_amount = capital * (risk_percent / 100)

        stop_distance = max(1.5 * atr, 1e-6)

        position_size = risk_amount / stop_distance

        if signal == "BUY":
            sl = price - stop_distance
            tp = price + 3 * atr
        else:
            sl = price + stop_distance
            tp = price - 3 * atr

        st.write("Position Size:", round(position_size, 3))
        st.write("Stop Loss:", round(sl, 5))
        st.write("Take Profit:", round(tp, 5))

    # Plot

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(data.index, data["Close"], label="Price")
    ax.plot(data.index, data["MA50"], label="MA50")
    ax.plot(data.index, data["MA200"], label="MA200")

    ax.legend()
    ax.grid()

    st.pyplot(fig)