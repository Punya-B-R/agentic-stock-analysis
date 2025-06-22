import streamlit as st
from agent import MarketAgent
import plotly.express as px
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import yfinance as yf
from advanced_recommender import AdvancedRecommender

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

                with st.expander("🧠 Agent Reasoning Log"):
                     for line in data["reasoning"]:
                        st.markdown(f"- {line}")
                
                # ===== Price Chart =====
                st.subheader("📊 Price Trends")
                # Prepare time series chart with Date index
                hist_df = pd.DataFrame({
                    "Date": pd.date_range(end=datetime.now(), periods=len(data.get("price_history", []))),
                    "Price": data.get("price_history", []),
                    "50-Day SMA": data.get("sma_50_history", []),
                    "200-Day SMA": data.get("sma_200_history", [])
                }).set_index("Date")

                # Melt dataframe to long format for plotly
                hist_df_long = hist_df.reset_index().melt(id_vars="Date", var_name="Metric", value_name="Value")

                # Plot with Plotly Express
                fig = px.line(
                    hist_df_long,
                    x="Date",
                    y="Value",
                    color="Metric",
                    labels={"Value": "Price ($)", "Date": "Date"}
                )
                st.plotly_chart(fig, use_container_width=True)
                info = yf.Ticker(ticker).info

                # Display them in two rows of columns
                row1 = st.columns(4)
                row1[0].metric("Open",       f"${info['open']:.2f}")
                row1[1].metric("Day Low",    f"${info['dayLow']:.2f}")
                row1[2].metric("Day High",   f"${info['dayHigh']:.2f}")
                row1[3].metric("Volume",     f"{info['volume']:,}")

                row2 = st.columns(2)
                row2[0].metric("52W Low",    f"${info['fiftyTwoWeekLow']:.2f}")
                row2[1].metric("52W High",   f"${info['fiftyTwoWeekHigh']:.2f}")
                
                # ===== News Section ==
               # Display News Section
                st.subheader("📰 Top 3 News Articles")

                if not data['news']:
                    st.warning("No news available")
                else:
                    for i, article in enumerate(data['news'][:3]):  # Only show top 3
                        if isinstance(article, dict):
                            # Create columns for title + link icon
                            col1, col2 = st.columns([0.9, 0.1])
                            
                            with col1:
                                st.markdown(f"**{i+1}. {article.get('title', 'Untitled News')}**")
                            
                            with col2:
                                if article.get('url'):
                                    st.markdown(
                                        f"[<img src='https://cdn-icons-png.flaticon.com/512/159/159828.png' width=20>]({article['url']})",
                                        unsafe_allow_html=True
                                    )
                            
                            # Publication date and summary
                            st.caption(f"🗓️ Published: {article.get('published_date', 'Date not available')}")
                            
                            # AI-generated summary (3-5 lines)
                            summary = article.get('summary', 
                                "No summary available. Click the link to read full article.")
                            st.write(summary)
                            
                            st.divider()
                        else:
                            st.warning(str(article))  # Display API errors
                
                # ===== AI Insights =====
                recommender = AdvancedRecommender()
                adv_analysis = recommender.recommend(ticker)
                st.subheader("🤖 Advanced AI Recommendation")
                st.write(adv_analysis)
                
                st.caption(f"Last updated: {datetime.fromisoformat(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            st.error(f"Critical error: {str(e)}")
            st.exception(e)
else:
    st.info("Enter a stock ticker and click Analyze to begin")

# Footer
st.divider()
st.caption("Powered by Mistral-7B, Tavily, and Streamlit")