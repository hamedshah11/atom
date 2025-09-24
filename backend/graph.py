from typing import TypedDict, Dict, Tuple
from langgraph.graph import StateGraph, START, END
from backend import agents

class IdeaState(TypedDict, total=False):
    idea: str
    plan: str
    market: str
    competition: str
    financial: str
    gtm: str
    risks: str
    critic: str
    final_report: str
    # Track which model actually answered each step
    models: Dict[str, str]

def _init_models(state: IdeaState):
    if "models" not in state or state["models"] is None:
        state["models"] = {}

def node_planner(state: IdeaState) -> Dict:
    _init_models(state)
    out, m = agents.planner_agent(state["idea"])
    state["models"]["planner"] = m
    return {"plan": out, "models": state["models"]}

def node_market(state: IdeaState) -> Dict:
    out, m = agents.market_analysis_agent(state["idea"])
    state["models"]["market"] = m
    return {"market": out, "models": state["models"]}

def node_competition(state: IdeaState) -> Dict:
    out, m = agents.competition_analysis_agent(state["idea"])
    state["models"]["competition"] = m
    return {"competition": out, "models": state["models"]}

def node_financial(state: IdeaState) -> Dict:
    out, m = agents.financial_feasibility_agent(state["idea"])
    state["models"]["financial"] = m
    return {"financial": out, "models": state["models"]}

def node_gtm(state: IdeaState) -> Dict:
    out, m = agents.gtm_strategy_agent(state["idea"])
    state["models"]["gtm"] = m
    return {"gtm": out, "models": state["models"]}

def node_risks(state: IdeaState) -> Dict:
    out, m = agents.risks_analysis_agent(state["idea"])
    state["models"]["risks"] = m
    return {"risks": out, "models": state["models"]}

def node_critic(state: IdeaState) -> Dict:
    analyses = {
        "Market": state.get("market", ""),
        "Competition": state.get("competition", ""),
        "Financial": state.get("financial", ""),
        "GTM": state.get("gtm", ""),
        "Risks": state.get("risks", "")
    }
    out, m = agents.critic_agent(state["idea"], analyses)
    state["models"]["critic"] = m
    return {"critic": out, "models": state["models"]}

def node_synth(state: IdeaState) -> Dict:
    analyses = {
        "Market": state.get("market", ""),
        "Competition": state.get("competition", ""),
        "Financial": state.get("financial", ""),
        "GTM": state.get("gtm", ""),
        "Risks": state.get("risks", "")
    }
    out, m = agents.synthesizer_agent(state["idea"], analyses, state.get("critic", ""))
    state["models"]["synth"] = m
    return {"final_report": out, "models": state["models"]}

def build_graph():
    g = StateGraph(IdeaState)
    g.add_node("planner", node_planner)
    g.add_node("market", node_market)
    g.add_node("competition", node_competition)
    g.add_node("financial", node_financial)
    g.add_node("gtm", node_gtm)
    g.add_node("risks", node_risks)
    g.add_node("critic", node_critic)
    g.add_node("synth", node_synth)

    g.add_edge(START, "planner")
    g.add_edge("planner", "market")
    g.add_edge("market", "competition")
    g.add_edge("competition", "financial")
    g.add_edge("financial", "gtm")
    g.add_edge("gtm", "risks")
    g.add_edge("risks", "critic")
    g.add_edge("critic", "synth")
    g.add_edge("synth", END)

    return g.compile()
