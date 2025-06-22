# langchain_agent.py
import os, json
from typing import Dict, Any
import yfinance as yf
from tavily import TavilyClient
from dotenv import load_dotenv
import google.generativeai as genai

# LangChain imports
from langchain.agents import initialize_agent, Tool
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# 1) Load keys & configure Gemini client
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2) Wrap Gemini as a LangChain LLM
class Gemini(LLM, BaseModel):
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.3)
    max_output_tokens: int = Field(default=1000)

    def _call(self, prompt: str, stop: list[str] | None = None) -> str:
        response = genai.generate_text(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return response.text

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        """Required by LangChain to know what type of LLM this is."""
        return "gemini"

# 3) Instantiate tools
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=TAVILY_KEY)

def fetch_stock(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="1y", auto_adjust=False)[["Close"]]
    if hist.empty:
        return json.dumps({"error":"No data"})
    return json.dumps({
        "closes": hist["Close"].tolist(),
        "sma50":  hist["Close"].rolling(50).mean().tolist(),
        "sma200": hist["Close"].rolling(200).mean().tolist(),
    })

def fetch_news(ticker: str) -> str:
    resp = tavily.search(query=f"{ticker} stock news", max_results=3, include_raw_content=True)
    articles = []
    for art in resp.get("results", [])[:3]:
        articles.append({
            "title":   art.get("title"),
            "content": art.get("content")
        })
    return json.dumps(articles)

def summarize_bullets(raw: str) -> str:
    prompt = (
        "Summarize the following in 3 bullet points:\n\n" + raw +
        "\n\nFormat:\n- Point 1\n- Point 2\n- Point 3"
    )
    # Directly call Gemini via our wrapper
    return Gemini().generate(prompt)

tools = [
    Tool(
        name="stock_data",
        func=fetch_stock,
        description="Returns JSON with closes, sma50, sma200 for a ticker."
    ),
    Tool(
        name="stock_news",
        func=fetch_news,
        description="Returns JSON list of top-3 {title,content} articles."
    ),
    Tool(
        name="summarize",
        func=summarize_bullets,
        description="Summarizes a text into 3 bullet points."
    ),
]

# 4) Build the agent with our Gemini LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_llm = Gemini()  
agent_chain = initialize_agent(
    tools,
    agent_llm,
    agent="zero-shot-react-description",
    verbose=False,
    memory=memory
)

# 5) Expose a simple runner
def run_agent(ticker: str) -> dict[str, Any]:
    prompt = (
        f"You are an AI market analyst for stock {ticker}. "
        "1) Call stock_data tool to get historical prices and SMAs. "
        "2) Analyze trend. If you need news, call stock_news and then summarize with summarize tool. "
        "3) Return JSON: {recommendation, key_points, price_insights, news_summary}."
    )
    raw = agent_chain.run(prompt)
    try:
        return json.loads(raw)
    except:
        return {"raw": raw}
