import os
from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Load API keys
dotenv_path = os.getenv('DOTENV_PATH', None)
if dotenv_path:
    from dotenv import load_dotenv as _ld
    _ld(dotenv_path)
else:
    load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel('gemini-2.0-flash')
generation_config = {"temperature": 0.4, "max_output_tokens": 800}


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return float(100 - (100 / (1 + rs.iloc[-1])))


def _compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {"macd": float(macd_line.iloc[-1]), "signal": float(signal_line.iloc[-1])}


def _compute_volatility(series: pd.Series, period: int = 20) -> float:
    return float(series.pct_change().rolling(window=period).std().iloc[-1] * 100)


class AdvancedRecommender:
    """
    Provides a free-form, advanced analysis and recommendation for a given stock ticker.
    """

    def recommend(self, ticker: str) -> str:
        # 1. Fetch historical data
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
        if hist.empty:
            return f"No historical data available for {ticker}."

        price = float(hist['Close'].iloc[-1])
        sma_50 = float(hist['Close'].rolling(window=50).mean().iloc[-1])
        sma_200 = float(hist['Close'].rolling(window=200).mean().iloc[-1])
        rsi = _compute_rsi(hist['Close'])
        macd_vals = _compute_macd(hist['Close'])
        volatility = _compute_volatility(hist['Close'])
        avg_vol = float(hist['Volume'].rolling(window=50).mean().iloc[-1])

        # 2. Build a dynamic prompt
        prompt = f"""
You are a senior market analyst with deep expertise.
Analyze the following metrics for {ticker} as of {datetime.now().strftime('%Y-%m-%d %H:%M')}:

- Current Price: ${price:.2f}
- 50-Day SMA:    ${sma_50:.2f}
- 200-Day SMA:   ${sma_200:.2f}
- RSI (14):      {rsi:.2f}
- MACD:          {macd_vals['macd']:.4f}
- Signal Line:   {macd_vals['signal']:.4f}
- Volatility (20d): {volatility:.2f}%
- Avg. Volume (50d): {avg_vol:,.0f}

Using these metrics, provide a comprehensive, free-form analysis that includes:
1. Market stance and context (bullish/bearish/neutral).
2. Detailed recommendation (Buy/Hold/Sell) with rationale.
3. Key risks or caveats.
4. Price targets or forecast ranges if appropriate.

Explain your reasoning in plain English, dive into which indicators matter most for this ticker, and tailor the depth to what an advanced trader would expect."""

        # 3. Call Gemini
        response = llm.generate_content(prompt, generation_config=generation_config)
        return response.text.strip()
