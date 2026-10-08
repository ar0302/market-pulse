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
        unit = "%" if name == "US 10Y Treasury" else ("$" if name == "Brent Crude Oil" else "")
        delta_str = f"{delta:+.2f} ({pct_delta:+.2f}%)"
        col.metric(label=name, value=f"{unit}{price:,.2f}", delta=delta_str)
    else:
        col.metric(label=name, value="N/A", delta="--")

st.markdown("---")

# ----------------- SECTION 2: NEWS & SENTIMENT ENGINE -----------------
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
            for entry in feed.entries[:8]:  # Limit entries per feed
                title = entry.get("title", "")
                link = entry.get("link", "#")
                pub_date = entry.get("published", entry.get("updated", "Just now"))
                
                # Sentiment score
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

# Calculate Sentiment Averages
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

# ----------------- SECTION 3: SENTIMENT METERS -----------------
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

# ----------------- SECTION 4: BREAKING NEWS TABS -----------------
st.subheader("📰 Breaking News & Headlines")
tab_ind, tab_glob = st.tabs(["🇮🇳 Indian Market Updates", "🌍 Global Market Updates"])

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

st.button("🔄 Refresh Data")
