from typing import TypedDict, Dict
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

def node_planner(state: IdeaState) -> Dict:
    return {"plan": agents.planner_agent(state["idea"])}

def node_market(state: IdeaState) -> Dict:
    return {"market": agents.market_analysis_agent(state["idea"])}

def node_competition(state: IdeaState) -> Dict:
    return {"competition": agents.competition_analysis_agent(state["idea"])}

def node_financial(state: IdeaState) -> Dict:
    return {"financial": agents.financial_feasibility_agent(state["idea"])}

def node_gtm(state: IdeaState) -> Dict:
    return {"gtm": agents.gtm_strategy_agent(state["idea"])}

def node_risks(state: IdeaState) -> Dict:
    return {"risks": agents.risks_analysis_agent(state["idea"])}

def node_critic(state: IdeaState) -> Dict:
    analyses = {
        "Market": state.get("market", ""),
        "Competition": state.get("competition", ""),
        "Financial": state.get("financial", ""),
        "GTM": state.get("gtm", ""),
        "Risks": state.get("risks", "")
    }
    return {"critic": agents.critic_agent(state["idea"], analyses)}

def node_synth(state: IdeaState) -> Dict:
    analyses = {
        "Market": state.get("market", ""),
        "Competition": state.get("competition", ""),
        "Financial": state.get("financial", ""),
        "GTM": state.get("gtm", ""),
        "Risks": state.get("risks", "")
    }
    return {"final_report": agents.synthesizer_agent(state["idea"], analyses, state.get("critic", ""))}

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
