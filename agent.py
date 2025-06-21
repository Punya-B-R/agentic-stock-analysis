from datetime import datetime
import yfinance
from tavily import TavilyClient
import openai
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()

class MarketAgent:
    def __init__(self):
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.llm = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            token=os.getenv("HF_API_KEY")  # HuggingFace API key
        )
        
    def analyze(self, ticker):
        """Analyze stock with technicals, news, and AI insights"""
        try:
            # ======================
            # 1. Fetch Stock Data
            # ======================
            stock = yfinance.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                raise ValueError(f"No data found for {ticker}")
                
            current_price = hist['Close'].iloc[-1]
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
            sma_200 = hist['Close'].rolling(200).mean().iloc[-1]
            
            # ======================
            # 2. Fetch News (with error fallback)
            # ======================
            try:
                news = self.tavily.search(
                    query=f"{ticker} stock news",
                    include_raw_content=False,
                    max_results=3
                )
                news_items = [
                    f"• {n['title']} ({n.get('source', 'Unknown')})" 
                    for n in news.get('results', [])[:3]
                ] or ["No recent news available"]
            except Exception as e:
                news_items = [f"News fetch failed: {str(e)}"]
            
            # ======================
            # 3. Generate AI Recommendation
            # ======================
            prompt = f"""
            Stock Analysis Request:
            - Ticker: {ticker}
            - Price: ${current_price:.2f}
            - 50-Day SMA: ${sma_50:.2f}
            - 200-Day SMA: ${sma_200:.2f}
            - Recent News:
            {chr(10).join(news_items)}

            Provide:
            1. Recommendation: [Buy/Hold/Sell]
            2. Key Reasons (3 bullet points max)
            3. Price Target Range (conservative/aggressive)
            """
            
            try:
                response = self.llm.text_generation(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.3,  # More deterministic output
                    stop_sequences=["\n\n"]  # Prevent rambling
                )
            except Exception as e:
                response = f"AI Error: {str(e)}"
            
            # ======================
            # 4. Format Results
            # ======================
            return {
                "ticker": ticker,
                "price": current_price,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "news": news_items,
                "analysis": self._parse_ai_response(response),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _parse_ai_response(self, text):
        """Extract structured data from Mistral's response"""
        return {
            "raw": text,
            "recommendation": (
                "Buy" if "buy" in text.lower() 
                else "Sell" if "sell" in text.lower() 
                else "Hold"
            ),
            "key_points": [
                line.strip() 
                for line in text.split('\n') 
                if line.startswith('- ') or line.startswith('• ')
            ][:3]  # Limit to top 3 points
        }