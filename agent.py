
from datetime import datetime
import yfinance as yf
from tavily import TavilyClient
import os
from typing import List, Dict, Union
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Union

load_dotenv()

class MarketAgent:
    def __init__(self):
        """Initialize with API clients"""
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Configure Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel('gemini-2.0-flash')
        self.generation_config = {
            "temperature": 0.3,
            "max_output_tokens": 1000,
        }

# inside your MarketAgent class...

    def analyze(self, ticker: str) -> Dict[str, Union[str, float, List]]:
        """End-to-end agentic pipeline with reasoning log."""
        reasoning: List[str] = []

        # Step 1: Fetch stock data
        reasoning.append(f"Step 1: Fetching 1y history for {ticker}")
        stock_data = self._get_stock_data(ticker)
        if "error" in stock_data:
            return {**stock_data, "reasoning": reasoning}

        price   = stock_data["current_price"]
        sma_50  = stock_data["sma_50"]
        sma_200 = stock_data["sma_200"]
        reasoning.append(f"  → Got price ${price:.2f}, SMA50 ${sma_50:.2f}, SMA200 ${sma_200:.2f}")

        # Step 2: Computing SMAs (already done above)
        reasoning.append("Step 2: SMAs computed")

        # Step 3: Ask Gemini if news is needed
        reasoning.append("Step 3: Asking Gemini if news is needed")
        need_news = self._decide_need_news(price, sma_50, sma_200)
        reasoning.append(f"  → Gemini says: {'Yes' if need_news else 'No'}")

        # Step 4: Conditionally fetch news
        if need_news:
            reasoning.append("Step 4: Fetching top-3 news articles")
            news_data = self._get_news(ticker)
            count = len(news_data.get("news", []))
            reasoning.append(f"  → Received {count} articles")
        else:
            reasoning.append("Step 4: Skipping news fetch")
            news_data = {"news": []}

        # Step 5: Summarize & analyze with Gemini
        reasoning.append("Step 5: Generating recommendation with Gemini")
        analysis = self._generate_analysis(
            ticker=ticker,
            price=price,
            sma_50=sma_50,
            sma_200=sma_200,
            news_items=news_data["news"],
        )
        verdict = analysis.get("recommendation", "N/A")
        reasoning.append(f"  → Recommendation: {verdict}")

        # Build final result
        result = {
            "ticker":         ticker,
            "price":          price,
            "sma_50":         sma_50,
            "sma_200":        sma_200,
            "news":           news_data.get("news", []),
            "analysis":       analysis,
            "timestamp":      datetime.now().isoformat(),
            "price_history":  stock_data.get("price_history", []),
            "sma_50_history": stock_data.get("sma_50_history", []),
            "sma_200_history":stock_data.get("sma_200_history", []),
            "reasoning":      reasoning,
        }

        return result


    def _get_stock_data(self, ticker: str) -> Dict:
        """Fetch price and technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                return {"error": f"No historical data found for {ticker}"}
            
            return {
                "ticker": ticker,
                "current_price": hist["Close"].iloc[-1],
                "sma_50": hist["Close"].rolling(50).mean().iloc[-1],
                "sma_200": hist["Close"].rolling(200).mean().iloc[-1],
                "price_history": hist["Close"].tolist(),
                "sma_50_history": hist["Close"].rolling(50).mean().tolist(),
                "sma_200_history": hist["Close"].rolling(200).mean().tolist()
            }
        except Exception as e:
            return {"error": f"Stock data error: {str(e)}"}

    def _get_news(self, ticker: str) -> Dict:
        """Fetch and summarize top 3 news articles"""
        try:
            news = self.tavily.search(
                query=f"{ticker} stock news",
                include_raw_content=True,
                include_domain=True,
                max_results=3
            )
            
            processed_news = []
            for article in news.get("results", [])[:3]:
                processed_news.append({
                    "title": article.get("title", "Untitled Article"),
                    "source": article.get("source", "Unknown"),
                    "url": article.get("url", ""),
                    "published_date": article.get("published_date", "Date not available"),
                    "content": article.get("content", ""),
                    "summary": self._summarize_article(article.get("content", ""))
                })
            
            return {"news": processed_news}
        except Exception as e:
            return {"error": f"News API error: {str(e)}"}

    def _summarize_article(self, text: str) -> str:
        """Generate 3-5 bullet point summary using Gemini"""
        if not text:
            return "No summary available"
            
        prompt = f"""
        Summarize this article in 3-5 concise bullet points:
        {text[:3000]}
        
        Format strictly as:
        - Point 1
        - Point 2
        - Point 3
        """
        
        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text.strip()
        except Exception as e:
            return f"Summary error: {str(e)}"

    def _generate_analysis(self, ticker: str, price: float, 
                         sma_50: float, sma_200: float,
                         news_items: List) -> Dict:
        """Generate AI recommendation with Gemini"""
        news_context = "\n".join(
            f"{i+1}. {n.get('title')} ({n.get('source')})" 
            for i, n in enumerate(news_items[:3]))
        
        prompt = f"""
        Stock Analysis Report for {ticker}:
        
        Current Price: ${price:.2f}
        50-Day SMA: ${sma_50:.2f}
        200-Day SMA: ${sma_200:.2f}
        
        Recent News:
        {news_context}
        
        Provide:
        1. Recommendation (Buy/Hold/Sell)
        2. 3 Key Reasons (bullet points)
        3. Price Targets (Conservative/Aggressive)
        
        Format exactly as:
        Recommendation: [Your Verdict]
        Reasons:
        - Reason 1
        - Reason 2
        - Reason 3
        Targets:
        - Conservative: $X
        - Aggressive: $Y
        """
        
        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return {
                "raw": response.text,
                "recommendation": self._extract_recommendation(response.text),
                "key_points": self._extract_bullet_points(response.text),
                "targets": self._extract_price_targets(response.text)
            }
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _decide_need_news(self, price: float, sma_50: float, sma_200: float) -> bool:
        """
        Ask Gemini: given price and SMAs, do we need recent news
        to make a better recommendation? Return True if yes.
        """
        prompt = f"""
    We have the following data for a stock:
    - Current Price: ${price:.2f}
    - 50-Day SMA: ${sma_50:.2f}
    - 200-Day SMA: ${sma_200:.2f}

    Question: Do you need recent news articles to give a better
    Buy/Hold/Sell recommendation? Answer ONLY "Yes" or "No".
    """

        try:
            resp = self.llm.generate_content(
                prompt,
                generation_config=self.generation_config
            ).text.strip().lower()
            return resp.startswith("yes")
        except:
            # on LLM failure, default to fetching news
            return True


    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation from Gemini response"""
        lines = text.split('\n')
        for line in lines:
            if "recommendation:" in line.lower():
                return line.split(':')[-1].strip()
        return "Hold"

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from Gemini response"""
        points = []
        in_reasons = False
        for line in text.split('\n'):
            if "reasons:" in line.lower():
                in_reasons = True
                continue
            if in_reasons and line.strip().startswith('-'):
                points.append(line.strip()[2:])
            if len(points) >= 3:
                break
        return points

    def _extract_price_targets(self, text: str) -> Dict:
        """Extract price targets from Gemini response"""
        targets = {"conservative": None, "aggressive": None}
        for line in text.split('\n'):
            if "conservative:" in line.lower():
                targets["conservative"] = line.split('$')[-1].strip()
            elif "aggressive:" in line.lower():
                targets["aggressive"] = line.split('$')[-1].strip()
        return targets
