# stock_agent_graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, List, Dict

# --- 1) Define the graph's state ---------
class StockState(TypedDict):
    ticker: str
    price: float
    recommendation: str
    key_points: List[str]

# --- 2) Reuse your MarketAgent -----------
from agent import MarketAgent
market_agent = MarketAgent()

# Node A: fetch + analyze
def fetch_and_analyze(state: StockState) -> StockState:
    result = market_agent.analyze(state["ticker"])
    return {
        "ticker": result["ticker"],
        "price":  result["price"],
        "recommendation": result["analysis"]["recommendation"],
        "key_points": result["analysis"]["key_points"],
    }

# Node B: (example) decide if alert needed
def decide_alert(state: StockState) -> StockState:
    # very simple rule: if recommendation is "Buy", mark
    if state["recommendation"].lower() == "buy":
        print(f"🔔 Alert: {state['ticker']} marked as BUY!")
    return state

# --- 3) Build the graph ------------------
saver = SqliteSaver("agent_graph.sqlite")
graph = StateGraph(StockState, saver=saver)

graph.add_node("analyze", fetch_and_analyze)
graph.add_node("alert_check", decide_alert)

graph.set_entry_point("analyze")
graph.add_edge("analyze", "alert_check")
graph.add_edge("alert_check", END)

# Compile once
compiled_graph = graph.compile()
