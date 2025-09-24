from typing import Dict, Iterator
from backend.graph import build_graph, IdeaState

compiled = build_graph()

def run_full(idea: str) -> IdeaState:
    state: IdeaState = {"idea": idea}
    return compiled.invoke(state)

def run_steps(idea: str) -> Iterator[Dict]:
    state: IdeaState = {"idea": idea}
    for update in compiled.stream(state):
        yield update
