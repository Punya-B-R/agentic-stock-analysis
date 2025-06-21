import streamlit as st
from agent import MarketAgent
import plotly.express as px
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize agent
agent = MarketAgent()

# Streamlit UI Config
st.set_page_config(layout="wide")
st.title("📈 AI Market Analyst Copilot")

# Sidebar for inputs
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Stock Ticker", "NVDA").upper()
    analyze_btn = st.button("Analyze", type="primary")

# Main Analysis Display
if analyze_btn and ticker:
    with st.spinner(f"Analyzing {ticker}..."):
        try:
            data = agent.analyze(ticker)
            
            if "error" in data:
                st.error(f"Analysis failed: {data['error']}")
            else:
                # ===== Price Overview =====
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${data['price']:.2f}")
                # 50-Day SMA
                col2.metric(
                    "50-Day SMA",
                    f"${data['sma_50']:.2f}",
                    delta=f"{(data['price'] - data['sma_50']) / data['sma_50'] * 100:.1f}%"
                )

                # 200-Day SMA
                col3.metric(
                    "200-Day SMA",
                    f"${data['sma_200']:.2f}",
                    delta=f"{(data['price'] - data['sma_200']) / data['sma_200'] * 100:.1f}%"
                )

                
                # ===== Price Chart =====
                st.subheader("📊 Price Trends")
                fig = px.line(
                    pd.DataFrame({
                        "Price": data.get('price_history', []),
                        "50-Day SMA": data.get('sma_50_history', []),
                        "200-Day SMA": data.get('sma_200_history', [])
                    }),
                    labels={"value": "Price ($)", "index": "Date"}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # ===== News Section =====
                st.subheader("📰 Latest News")
                if not data['news']:
                    st.warning("No news available")
                else:
                    for article in data['news']:
                        if isinstance(article, dict):
                            with st.expander(article.get('title', 'Untitled')):
                                st.caption(f"Source: {article.get('source', 'Unknown')}")
                                st.write(article.get('content', 'No content available')[:500] + "...")
                                if article.get('url'):
                                    st.markdown(f"[Read more]({article['url']})")
                        else:
                            st.warning(str(article))  # Display API errors
                
                # ===== AI Insights =====
                st.subheader("🤖 AI Recommendation")
                if isinstance(data['analysis'], dict):
                    st.success(f"Verdict: **{data['analysis'].get('recommendation', 'N/A')}**")
                    st.markdown("**Key Points:**")
                    for point in data['analysis'].get('key_points', []):
                        st.write(f"- {point}")
                    st.text_area("Full Analysis", 
                                value=data['analysis'].get('raw', ''), 
                                height=150)
                else:
                    st.warning(str(data['analysis']))
                
                st.caption(f"Last updated: {datetime.fromisoformat(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.exception(e)
else:
    st.info("Enter a stock ticker and click Analyze to begin")

# Footer
st.divider()
st.caption("Powered by Mistral-7B, Tavily, and Streamlit")