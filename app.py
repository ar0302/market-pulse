import streamlit as st
import yfinance as yf
import feedparser
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure sentiment lexicon is available
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Indian Market Pulse", page_icon="📈", layout="centered")

st.title("🇮🇳 Indian Market Pulse & Sentiment")
st.caption("Live NIFTY, Bank NIFTY, India VIX & News Sentiment Analysis")

# --- 1. Fetch Indian Indices ---
@st.cache_data(ttl=300)
def fetch_indian_indices():
    tickers = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "India VIX": "^INDIAVIX"
    }
    data = {}
    for name, sym in tickers.items():
        try:
            hist = yf.Ticker(sym).history(period="5d")
            if len(hist) >= 2:
                latest = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((latest - prev) / prev) * 100
                data[name] = (latest, change)
        except Exception:
            continue
    return data

indices = fetch_indian_indices()

# Display Indices
cols = st.columns(3)
for i, (name, val_chg) in enumerate(indices.items()):
    val, chg = val_chg
    with cols[i]:
        st.metric(label=name, value=f"{val:,.2f}", delta=f"{chg:+.2f}%")

st.divider()

# --- 2. Fetch Breaking Indian Market News ---
@st.cache_data(ttl=600)
def fetch_indian_news():
    # Economic Times & Moneycontrol Market RSS Feeds
    feed_urls = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/MCtopnews.xml"
    ]
    news_items = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:8]:
            score = sia.polarity_scores(entry.title)['compound']
            news_items.append({
                "title": entry.title,
                "link": entry.link,
                "score": score
            })
    return news_items

news_data = fetch_indian_news()

# Calculate Overall News Sentiment
if news_data:
    avg_score = sum(item['score'] for item in news_data) / len(news_data)
    
    st.subheader("Market Sentiment Meter")
    if avg_score > 0.15:
        st.success(f"🟢 **Bullish Sentiment** (Score: {avg_score:+.2f}) — News flow is predominantly positive.")
    elif avg_score < -0.15:
        st.error(f"🔴 **Bearish Sentiment** (Score: {avg_score:+.2f}) — News flow indicates cautious/negative tone.")
    else:
        st.info(f"⚪ **Neutral Sentiment** (Score: {avg_score:+.2f}) — Mixed or balanced headlines.")

    # Volatility Check
    if "India VIX" in indices and indices["India VIX"][0] > 18:
        st.warning("⚠️ **High Volatility Alert:** India VIX is elevated. Expect sharp intraday swings.")

    st.divider()
    st.subheader("Breaking Market Headlines")
    
    for item in news_data[:10]:
        tag = "🟢" if item['score'] > 0.1 else ("🔴" if item['score'] < -0.1 else "⚪")
        st.markdown(f"{tag} [{item['title']}]({item['link']})")
else:
    st.info("Unable to fetch live news feeds at the moment.")
