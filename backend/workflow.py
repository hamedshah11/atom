from langgraph.graph import StateGraph, State, field, START, END
from backend import agents

# Define the state structure for the analysis
class IdeaAnalysisState(State):
    idea: str = field()
    plan: str = field(default_factory=str)
    market_analysis: str = field(default_factory=str)
    competition_analysis: str = field(default_factory=str)
    financial_analysis: str = field(default_factory=str)
    gtm_strategy: str = field(default_factory=str)
    risks_analysis: str = field(default_factory=str)
    critic_feedback: str = field(default_factory=str)
    final_report: str = field(default_factory=str)

# Define node functions for each step, using the agent functions from agents.py
def planner_node(state: IdeaAnalysisState):
    plan_text = agents.planner_agent(state.idea)
    return {"plan": plan_text}

def market_node(state: IdeaAnalysisState):
    analysis_text = agents.market_analysis_agent(state.idea)
    return {"market_analysis": analysis_text}

def competition_node(state: IdeaAnalysisState):
    analysis_text = agents.competition_analysis_agent(state.idea)
    return {"competition_analysis": analysis_text}

def financial_node(state: IdeaAnalysisState):
    analysis_text = agents.financial_feasibility_agent(state.idea)
    return {"financial_analysis": analysis_text}

def gtm_node(state: IdeaAnalysisState):
    analysis_text = agents.gtm_strategy_agent(state.idea)
    return {"gtm_strategy": analysis_text}

def risks_node(state: IdeaAnalysisState):
    analysis_text = agents.risks_analysis_agent(state.idea)
    return {"risks_analysis": analysis_text}

def critic_node(state: IdeaAnalysisState):
    # Prepare a dict of analyses to feed the critic agent
    analyses = {
        "Market": state.market_analysis,
        "Competition": state.competition_analysis,
        "Financial": state.financial_analysis,
        "GTM": state.gtm_strategy,
        "Risks": state.risks_analysis
    }
    feedback_text = agents.critic_agent(state.idea, analyses)
    return {"critic_feedback": feedback_text}

def synthesizer_node(state: IdeaAnalysisState):
    analyses = {
        "Market": state.market_analysis,
        "Competition": state.competition_analysis,
        "Financial": state.financial_analysis,
        "GTM": state.gtm_strategy,
        "Risks": state.risks_analysis
    }
    final_report_text = agents.synthesizer_agent(state.idea, analyses, state.critic_feedback)
    return {"final_report": final_report_text}

# Build the LangGraph stateful workflow
def build_analysis_graph():
    graph = StateGraph(IdeaAnalysisState)
    graph.add_node("planner", planner_node)
    graph.add_node("market", market_node)
    graph.add_node("competition", competition_node)
    graph.add_node("financial", financial_node)
    graph.add_node("gtm", gtm_node)
    graph.add_node("risks", risks_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    # Define edges to form the sequence of analysis
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "market")
    graph.add_edge("market", "competition")
    graph.add_edge("competition", "financial")
    graph.add_edge("financial", "gtm")
    graph.add_edge("gtm", "risks")
    graph.add_edge("risks", "critic")
    graph.add_edge("critic", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()
