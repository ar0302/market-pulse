import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
st.caption("Real-time Indices, Commodities, Bond Yields, Live News & Sentiment Analysis")

# ----------------- SECTION 1: MARKET TICKERS -----------------
st.subheader("📊 Key Market Gauges")

TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Brent Crude Oil": "BZ=F",
    "US 10Y Treasury": "^TNX",
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
                delta = 0.0
                pct_delta = 0.0
            else:
                current_price, delta, pct_delta = None, None, None
            metrics[name] = (current_price, delta, pct_delta)
        except Exception:
            metrics[name] = (None, None, None)
    return metrics

market_data = fetch_market_metrics()

cols = st.columns(len(TICKERS))
for col, (name, val) in zip(cols, market_data.items()):
    price, delta, pct_delta = val
    if price is not None:
        unit = "%" if name == "US 10Y Treasury" else ("$" if name in ["Brent Crude Oil", "Gold", "Silver"] else "")
        delta_str = f"{delta:+.2f} ({pct_delta:+.2f}%)"
        col.metric(label=name, value=f"{unit}{price:,.2f}", delta=delta_str)
    else:
        col.metric(label=name, value="N/A", delta="--")

st.markdown("---")

# ----------------- SECTION 2: TOP GAINERS & LOSERS -----------------
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
                "Symbol": [s.replace(".NS", "") for s in pct_change.index],
                "Price (₹)": curr_close.values.round(2),
                "Change (%)": pct_change.values.round(2)
            }).dropna()
            
            gainers = df.sort_values(by="Change (%)", ascending=False).head(5).reset_index(drop=True)
            losers = df.sort_values(by="Change (%)", ascending=True).head(5).reset_index(drop=True)
            return gainers, losers
    except Exception:
        pass
    return pd.DataFrame(), pd.DataFrame()

# ----------------- SECTION 3: NEWS & SENTIMENT ENGINE -----------------
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
                    label = "Bullish / Positive"
                    color = "🟢"
                elif score <= -0.05:
                    label = "Bearish / Negative"
                    color = "🔴"
                else:
                    label = "Neutral"
                    color = "⚪"
                    
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

# ----------------- SECTION 4: SENTIMENT METERS -----------------
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

# ----------------- SECTION 5: TABS (NEWS + GAINERS/LOSERS) -----------------
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
            st.dataframe(
                gainers_df.style.format({"Price (₹)": "₹{:,.2f}", "Change (%)": "+{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Unable to fetch gainers data at the moment.")
            
    with col_lose:
        st.error("### 🔴 Top 5 Losers")
        if not losers_df.empty:
            st.dataframe(
                losers_df.style.format({"Price (₹)": "₹{:,.2f}", "Change (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Unable to fetch losers data at the moment.")

st.button("🔄 Refresh Data")
