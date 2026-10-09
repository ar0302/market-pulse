import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

# ----------------- NLTK SETUP -----------------
@st.cache_resource
def load_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_vader()

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Global & Indian Market Pulse",
    page_icon="📈",
    layout="wide"
)

st.title("🌐 Global & Indian Macro Dashboard")
st.caption("Real-time Indices, Reliable Indian Bullion Rates, Live News & Sentiment Analysis")

# ----------------- SECTION 1: RELIABLE BULLION FETCHER -----------------
@st.cache_data(ttl=300)
def fetch_live_indian_bullion():
    headers = {"User-Agent": "Mozilla/5.0"}
    gold_price, gold_delta = None, None
    silver_price, silver_delta = None, None

    # GoodReturns / Financial portal se official Indian domestic rates
    try:
        url = "https://www.goodreturns.in/gold-rates/"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # 24K 10g Gold spot price element
            gold_elem = soup.find("strong", id="el_gold_24k_10gm") or soup.find("div", class_="gold_silver_rates")
            if gold_elem:
                text = gold_elem.text.replace("₹", "").replace(",", "").strip()
                val = float(''.join(c for c in text.split()[0] if c.isdigit() or c == '.'))
                gold_price = val
    except Exception:
        pass

    try:
        url_silver = "https://www.goodreturns.in/silver-rates/"
        resp_s = requests.get(url_silver, headers=headers, timeout=5)
        if resp_s.status_code == 200:
            soup_s = BeautifulSoup(resp_s.text, "html.parser")
            silver_elem = soup_s.find("strong", id="el_silver_1000gm") or soup_s.find("div", class_="gold_silver_rates")
            if silver_elem:
                text = silver_elem.text.replace("₹", "").replace(",", "").strip()
                val = float(''.join(c for c in text.split()[0] if c.isdigit() or c == '.'))
                silver_price = val
    except Exception:
        pass

    # Fallback agar scraper block ho: Nippon Gold/Silver units
    if not gold_price:
        try:
            g_hist = yf.Ticker("GOLDBEES.NS").history(period="2d")
            if len(g_hist) >= 2:
                # 1 unit GoldBEES = ~0.01g ya exact weight formula MCX Parity
                curr_g = g_hist['Close'].iloc[-1]
                prev_g = g_hist['Close'].iloc[-2]
                gold_price = curr_g * 1000
                gold_delta = (curr_g - prev_g) * 1000
        except Exception:
            pass

    if not silver_price:
        try:
            s_hist = yf.Ticker("SILVERBEES.NS").history(period="2d")
            if len(s_hist) >= 2:
                curr_s = s_hist['Close'].iloc[-1]
                prev_s = s_hist['Close'].iloc[-2]
                silver_price = curr_s * 1000
                silver_delta = (curr_s - prev_s) * 1000
        except Exception:
            pass

    return gold_price, gold_delta, silver_price, silver_delta

# ----------------- SECTION 2: MARKET TICKERS -----------------
st.subheader("📊 Key Market Gauges")

TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "Brent Crude ($)": "BZ=F",
    "US 10Y Yield": "^TNX",
    "India VIX": "^INDIAVIX"
}

@st.cache_data(ttl=120)
def fetch_market_metrics():
    metrics = {}
    for name, sym in TICKERS.items():
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2]
                delta = current_price - prev_price
                pct_delta = (delta / prev_price) * 100
            elif len(hist) == 1:
                current_price = hist['Close'].iloc[-1]
                delta, pct_delta = 0.0, 0.0
            else:
                current_price, delta, pct_delta = None, None, None
            metrics[name] = (current_price, delta, pct_delta)
        except Exception:
            metrics[name] = (None, None, None)
    return metrics

market_data = fetch_market_metrics()
gold_p, gold_d, silver_p, silver_d = fetch_live_indian_bullion()

# Display row
cols = st.columns(len(TICKERS) + 2)

# Index 0 & 1: Indices
for i, name in enumerate(["NIFTY 50", "SENSEX"]):
    price, delta, pct_delta = market_data.get(name, (None, None, None))
    if price is not None:
        cols[i].metric(label=name, value=f"{price:,.2f}", delta=f"{delta:+.2f} ({pct_delta:+.2f}%)")
    else:
        cols[i].metric(label=name, value="N/A", delta="--")

# Index 2: Gold 24K (10g)
if gold_p:
    delta_text = f"{gold_d:+.2f}" if gold_d else "Live"
    cols[2].metric(label="Gold 24K (10g / INR)", value=f"₹{gold_p:,.0f}", delta=delta_text)
else:
    cols[2].metric(label="Gold 24K (10g / INR)", value="N/A", delta="--")

# Index 3: Silver (1kg)
if silver_p:
    delta_text_s = f"{silver_d:+.2f}" if silver_d else "Live"
    cols[3].metric(label="Silver (1kg / INR)", value=f"₹{silver_p:,.0f}", delta=delta_text_s)
else:
    cols[3].metric(label="Silver (1kg / INR)", value="N/A", delta="--")

# Remaining Gauges
rem_names = ["Brent Crude ($)", "US 10Y Yield", "India VIX"]
for idx, name in enumerate(rem_names, start=4):
    price, delta, pct_delta = market_data.get(name, (None, None, None))
    if price is not None:
        unit = "%" if name == "US 10Y Yield" else ("$" if "$" in name else "")
        cols[idx].metric(label=name, value=f"{unit}{price:,.2f}", delta=f"{delta:+.2f} ({pct_delta:+.2f}%)")
    else:
        cols[idx].metric(label=name, value="N/A", delta="--")

st.markdown("---")

# ----------------- SECTION 3: TOP GAINERS & LOSERS -----------------
NIFTY_50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "M&M.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
    "NTPC.NS", "POWERGRID.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ONGC.NS",
    "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS",
    "TECHM.NS", "HINDALCO.NS", "GRASIM.NS", "CIPLA.NS", "TATAMOTORS.NS",
    "BPCL.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "APOLLOHOSP.NS",
    "DIVISLAB.NS", "BRITANNIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS",
    "SHRIRAMFIN.NS", "TRENT.NS", "BEL.NS", "NESTLEIND.NS", "INDUSINDBK.NS"
]

@st.cache_data(ttl=180)
def fetch_top_movers():
    try:
        data = yf.download(NIFTY_50_STOCKS, period="2d", interval="1d", progress=False)["Close"]
        if len(data) >= 2:
            prev_close = data.iloc[-2]
            curr_close = data.iloc[-1]
            pct_change = ((curr_close - prev_close) / prev_close) * 100
            
            df = pd.DataFrame({
                "Symbol": pct_change.index,
                "Clean_Symbol": [s.replace(".NS", "") for s in pct_change.index],
                "Price (₹)": curr_close.values.round(2),
                "Change (%)": pct_change.values.round(2)
            }).dropna()
            
            gainers = df.sort_values(by="Change (%)", ascending=False).head(5).reset_index(drop=True)
            losers = df.sort_values(by="Change (%)", ascending=True).head(5).reset_index(drop=True)
            return gainers, losers
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()

# ----------------- SECTION 4: NEWS & SENTIMENT ENGINE -----------------
FEEDS = {
    "Indian Market": [
        {"source": "Economic Times - Markets", "url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"},
        {"source": "Economic Times - Economy", "url": "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms"}
    ],
    "Global Market": [
        {"source": "Reuters Business", "url": "https://feeds.feedburner.com/reuters/businessNews"},
        {"source": "CNBC International", "url": "https://search.cnbc.com/rs/search/combinedlist.do?keywords=world%20markets&partnerId=wrss01&strip=0"}
    ]
}

@st.cache_data(ttl=300)
def fetch_news_and_sentiment():
    results = {"Indian Market": [], "Global Market": []}
    for category, feed_list in FEEDS.items():
        for feed_info in feed_list:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:8]:
                title = entry.get("title", "")
                link = entry.get("link", "#")
                pub_date = entry.get("published", entry.get("updated", "Just now"))
                score = sia.polarity_scores(title)["compound"]
                
                if score >= 0.05:
                    label, color = "Bullish / Positive", "🟢"
                elif score <= -0.05:
                    label, color = "Bearish / Negative", "🔴"
                else:
                    label, color = "Neutral", "⚪"
                    
                results[category].append({
                    "title": title,
                    "link": link,
                    "source": feed_info["source"],
                    "published": pub_date,
                    "score": score,
                    "label": label,
                    "badge": f"{color} {label}"
                })
    return results

news_data = fetch_news_and_sentiment()

def compute_sentiment_stats(items):
    if not items:
        return 0.0, "Neutral"
    scores = [item["score"] for item in items]
    avg_score = sum(scores) / len(scores)
    if avg_score > 0.08:
        sentiment = "Bullish / Optimistic"
    elif avg_score < -0.08:
        sentiment = "Bearish / Cautious"
    else:
        sentiment = "Neutral / Mixed"
    return avg_score, sentiment

ind_score, ind_sentiment = compute_sentiment_stats(news_data["Indian Market"])
glob_score, glob_sentiment = compute_sentiment_stats(news_data["Global Market"])

# ----------------- SECTION 5: SENTIMENT METERS -----------------
st.subheader("🧭 Market Sentiment Gauges")
sent_col1, sent_col2 = st.columns(2)

def create_gauge(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [-1, -0.1], 'color': '#ffb3b3'},
                {'range': [-0.1, 0.1], 'color': '#f0f2f6'},
                {'range': [0.1, 1], 'color': '#b3ffb3'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=40, b=20))
    return fig

with sent_col1:
    st.markdown(f"**India Market Sentiment:** `{ind_sentiment}`")
    st.plotly_chart(create_gauge(ind_score, "Indian Sentiment (-1 to +1)"), use_container_width=True)

with sent_col2:
    st.markdown(f"**Global Market Sentiment:** `{glob_sentiment}`")
    st.plotly_chart(create_gauge(glob_score, "Global Sentiment (-1 to +1)"), use_container_width=True)

st.markdown("---")

# ----------------- SECTION 6: TABS (NEWS + GAINERS/LOSERS + CHART) -----------------
st.subheader("📰 Market Insights & Data")
tab_ind, tab_glob, tab_movers = st.tabs([
    "🇮🇳 Indian Market Updates", 
    "🌍 Global Market Updates", 
    "🚀 Top Gainers & Losers (Nifty 50)"
])

with tab_ind:
    for news in news_data["Indian Market"]:
        with st.container(border=True):
            st.markdown(f"**[{news['title']}]({news['link']})**")
            st.caption(f"Source: {news['source']} | Date: {news['published']} | Sentiment: {news['badge']}")

with tab_glob:
    for news in news_data["Global Market"]:
        with st.container(border=True):
            st.markdown(f"**[{news['title']}]({news['link']})**")
            st.caption(f"Source: {news['source']} | Date: {news['published']} | Sentiment: {news['badge']}")

with tab_movers:
    gainers_df, losers_df = fetch_top_movers()
    col_gain, col_lose = st.columns(2)
    
    with col_gain:
        st.success("### 🟢 Top 5 Gainers")
        if not gainers_df.empty:
            show_df = gainers_df[["Clean_Symbol", "Price (₹)", "Change (%)"]].rename(columns={"Clean_Symbol": "Stock"})
            st.dataframe(
                show_df.style.format({"Price (₹)": "₹{:,.2f}", "Change (%)": "+{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Gainers data fetch nahi ho paya.")
            
    with col_lose:
        st.error("### 🔴 Top 5 Losers")
        if not losers_df.empty:
            show_df = losers_df[["Clean_Symbol", "Price (₹)", "Change (%)"]].rename(columns={"Clean_Symbol": "Stock"})
            st.dataframe(
                show_df.style.format({"Price (₹)": "₹{:,.2f}", "Change (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Losers data fetch nahi ho paya.")

    # Candlestick Stock Chart
    st.markdown("---")
    st.markdown("### 📈 Stock Technical Chart")
    
    combined_stocks = []
    if not gainers_df.empty:
        combined_stocks.extend(gainers_df["Symbol"].tolist())
    if not losers_df.empty:
        combined_stocks.extend(losers_df["Symbol"].tolist())
        
    if combined_stocks:
        col_select, col_period = st.columns([3, 1])
        with col_select:
            selected_sym = st.selectbox(
                "Stock select karein:",
                options=combined_stocks,
                format_func=lambda x: x.replace(".NS", "")
            )
        with col_period:
            period_choice = st.selectbox("Timeframe:", options=["1mo", "3mo", "6mo", "1y"], index=1)
            
        if selected_sym:
            stock_data = yf.Ticker(selected_sym).history(period=period_choice)
            if not stock_data.empty:
                chart_fig = go.Figure()
                chart_fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="Price"
                ))
                chart_fig.update_layout(
                    title=f"{selected_sym.replace('.NS', '')} ({period_choice}) Candlestick Chart",
                    yaxis_title="Price (₹)",
                    xaxis_rangeslider_visible=False,
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(chart_fig, use_container_width=True)

st.button("🔄 Refresh Data")
