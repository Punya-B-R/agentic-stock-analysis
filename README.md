# 📈 Agentic Stock Analyst Copilot

An agentic AI-powered application that delivers real-time stock insights, technical indicator analysis, intelligent news summarization, and LLM-driven trading recommendations. At its core, this tool acts as a fully autonomous Market Research Agent — analyzing data, interpreting trends, summarizing external context, and forming its own conclusions, just like a human financial analyst.

## Agentic Design 

This application does more than just present data — it operates as an **autonomous AI agent** capable of making informed trading recommendations. Here's how it works at an agentic level:

| **Agentic Behavior**   | **Implementation**                                                                 |
|------------------------|------------------------------------------------------------------------------------|
| **Observation**        | Collects and monitors real-time market data (price, volume, technical indicators)  |
| **Contextual Awareness** | Uses Tavily AI to incorporate external context through real-time news summarization |
| **Reasoning**          | Applies financial analysis using Gemini LLM to interpret market sentiment           |
| **Action Suggestion**  | Suggests a course of action (Buy / Hold / Sell) with rationale                     |
| **Adaptability**       | Updates recommendations based on evolving market and news data                     |

Unlike fixed rule-based tools, this copilot uses reasoning to support deeper, more adaptive equity analysis.

## Dashboard Preview
![Alt Text](https://github.com/Punya-B-R/agentic-stock-analyst-copilot/blob/main/assets/picture.png?raw=true)

## 🔍 Features

- Real-time price and technical indicators: **SMA (50/200)**, **RSI (14)**, **MACD**, **Volatility**
- Intelligent news summarization using **Tavily AI**
- Natural-language reasoning and insights powered by **Google Gemini**
- Final actionable recommendation: **Buy / Hold / Sell**
- Transparent agent reasoning log
- Interactive trend visualization via **Plotly**
- Fully autonomous — no manual inputs required

---

## Indicators Used

| **Metric**     | **Use**                                                 |
|----------------|----------------------------------------------------------|
| **RSI (14)**   | Detects momentum by measuring overbought/oversold pressure using the last 14 periods |
| **MACD**       | Identifies trend-following momentum and potential reversals |
| **SMA (50/200)** | Highlights medium and long-term price trends             |
| **Volatility** | Gauges risk by calculating percentage deviation in recent price movements |

## Local Setup Guide
### 1. **Install Python 3.9+**  
   Download from [python.org](https://www.python.org/downloads/) then verify:
   ```bash
   python --version
   ```

### 2. Clone the Repository
   ```bash
   git clone https://github.com/Punya-B-R/agentic-stock-analyst-copilot.git
   cd agentic-stock-analyst-copilot
   ```

### 3. Set Up Virtual Environment
   ```bash
   python -m venv venv
   venv\Scripts\activate # Windows
   source venv/bin/activate # macOS/Linux
   ```

### 4. Install Dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
### 5. Configure API Key
   - In project root, create an environment file.
   - Get key from [Google AI Studio](https://aistudio.google.com/app/apikey) and [Tavily](https://app.tavily.com/home)
   - Add to `.env`:
     ```text
     GEMINI_API_KEY="YOUR_API_KEY"  
     TAVILY_API_KEY="YOUR_API_KEY"
     ```
