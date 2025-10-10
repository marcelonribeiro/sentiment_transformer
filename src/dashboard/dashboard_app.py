import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import plotly.express as px

# Configuration
from src.config import settings

st.set_page_config(page_title="Sentiment Thermometer", layout="wide")

@st.cache_data(ttl=600)
def get_summary_data():
    try:
        res = requests.get(f"{settings.API_BASE_URL}/sentiments/summary")
        if res.ok:
            return pd.DataFrame(res.json())
        else:
            st.error(f"Failed to fetch summary data. Status: {res.status_code}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the API. Is it running? Error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_ticker_timeseries(ticker):
    res = requests.get(f"{settings.API_BASE_URL}/sentiment/{ticker}/timeseries")
    return pd.DataFrame(res.json()) if res.ok else pd.DataFrame()

@st.cache_data(ttl=600)
def get_ticker_news(ticker):
    res = requests.get(f"{settings.API_BASE_URL}/sentiment/{ticker}/news")
    return pd.DataFrame(res.json()) if res.ok else pd.DataFrame()


# Main Dashboard Page
st.title("📈 Stock Sentiment Thermometer")

summary_df = get_summary_data()

if not summary_df.empty:
    st.header("Overall Market Sentiment")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 15 Most Positive")
        top_15 = summary_df.head(15)
        fig = px.bar(top_15, x='avg_sentiment', y='ticker', orientation='h',
                     color='avg_sentiment', color_continuous_scale='Greens')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 Most Negative")
        bottom_15 = summary_df.tail(15).sort_values('avg_sentiment', ascending=True)
        fig = px.bar(bottom_15, x='avg_sentiment', y='ticker', orientation='h',
                     color='avg_sentiment', color_continuous_scale='Reds_r')
        fig.update_layout(yaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)

# Ticker Detail Page
st.header("Detailed Ticker Analysis")
if not summary_df.empty:
    all_tickers = [""] + sorted(summary_df['ticker'].tolist())
    selected_ticker = st.selectbox("Select a Ticker to Analyze", options=all_tickers)

    if selected_ticker:
        # Fetch the single, most recent record for the selected ticker
        ticker_data_res = requests.get(f"{settings.API_BASE_URL}/sentiment/{selected_ticker}")
        if ticker_data_res.ok:
            ticker_data = ticker_data_res.json()

            # Gauge chart
            st.subheader(f"Current Sentiment for {selected_ticker}")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ticker_data['avg_sentiment'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "3-Month Rolling Avg. (0=Neg, 0.5=Neu, 1=Pos)"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.3], 'color': "#d9534f"},    # Negative (Red)
                           {'range': [0.3, 0.5], 'color': "lightgray"}, # Neutral (Gray)
                           {'range': [0.5, 1], 'color': "#5cb85c"}]    # Positive (Green)
                       }))
            st.plotly_chart(fig, use_container_width=True)

            # Time series chart
            st.subheader("Sentiment Over Time (3-Month Rolling Average)")
            ts_df = get_ticker_timeseries(selected_ticker)
            if not ts_df.empty:
                fig = px.line(
                    ts_df,
                    x='calculation_date',
                    y='avg_sentiment',
                    title="Daily Rolling Sentiment",
                    markers=True,
                    template="plotly_white",
                    labels={
                        "calculation_date": "Date",
                        "avg_sentiment": "Sentiment Score"
                    }
                )
                fig.update_traces(marker=dict(size=8), line=dict(width=3))
                fig.update_layout(xaxis_title=None, yaxis_title="Sentiment Score")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No time series data available.")

            # News table
            st.subheader("Recent News Influencing the Score")
            news_df = get_ticker_news(selected_ticker)
            if not news_df.empty:
                st.dataframe(
                    news_df[['news_publication_date', 'news_title', 'sentiment_score', 'news_link']],
                    column_config={
                        "news_link": st.column_config.LinkColumn(
                            "Link",
                            help="Click to read the full article",
                            display_text="Read Article"
                        ),
                        "sentiment_score": st.column_config.NumberColumn(
                            "Sentiment",
                            format="%.3f"
                        )
                    },
                    hide_index=True
                )
            else:
                st.write("No recent news found for this ticker.")