import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Macro Pulse", layout="centered")

st.title("Macro Market Indicator")
st.caption("Live Treasury Yields, US Dollar Index, and S&P 500 signals")

@st.cache_data(ttl=300)
def fetch_market_data():
    tickers = {
        "10Y Yield": "^TNX",
        "US Dollar (DXY)": "DX-Y.NYB",
        "S&P 500": "^GSPC",
        "VIX (Volatility)": "^VIX"
    }
    data = {}
    for name, sym in tickers.items():
        hist = yf.Ticker(sym).history(period="5d")
        if len(hist) >= 2:
            latest = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((latest - prev) / prev) * 100
            data[name] = (latest, change)
    return data

try:
    metrics = fetch_market_data()
    cols = st.columns(2)
    i = 0
    for name, (val, chg) in metrics.items():
        with cols[i % 2]:
            st.metric(label=name, value=f"{val:.2f}", delta=f"{chg:+.2f}%")
        i += 1

    st.divider()
    st.subheader("Market Posture Summary")
    
    dxy_chg = metrics.get("US Dollar (DXY)", (0, 0))[1]
    tnx_chg = metrics.get("10Y Yield", (0, 0))[1]
    vix_val = metrics.get("VIX (Volatility)", (0, 0))[0]

    # Basic macro logic heuristic
    if vix_val > 25:
        st.error("⚠️ High Volatility Regime: Defensive posture favored.")
    elif dxy_chg > 0.5 and tnx_chg > 0.5:
        st.warning("⚠️ Rising Rates & Strong Dollar: Downward pressure on risk assets.")
    elif dxy_chg < -0.3 and tnx_chg < -0.3:
        st.success("🟢 Easing Liquidity: Favorable tailwinds for equities.")
    else:
        st.info("ℹ️ Neutral/Consolidation: Mixed macro signals.")

except Exception as e:
    st.error(f"Error fetching live data: {e}")
