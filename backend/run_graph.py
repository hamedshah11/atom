from typing import Dict, Iterator
from backend.graph import build_graph, IdeaState

compiled = build_graph()

def run_full(idea: str, region: str = "Pakistan") -> IdeaState:
    """Run full analysis with region support"""
    state: IdeaState = {"idea": idea, "region": region}
    return compiled.invoke(state)

def run_steps(idea: str, region: str = "Pakistan") -> Iterator[Dict]:
    """Run analysis step by step with region support"""
    state: IdeaState = {"idea": idea, "region": region}
    for update in compiled.stream(state):
        yield update
