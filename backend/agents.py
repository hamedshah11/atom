import os
from typing import Dict, Optional
from openai import OpenAI

# Default model (override via env/Streamlit secrets: MODEL_ALL="gpt-5-mini" etc.)
MODEL_ALL = os.getenv("MODEL_ALL", "gpt-5")

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _respond(
    instructions: str,
    prompt: str,
    *,
    model: str = MODEL_ALL,
    effort: str = "medium",          # minimal | low | medium | high
    verbosity: str | None = None,    # low | medium | high
    max_tokens: int = 1200,
) -> str:
    """
    Unified Responses API call (GPT-5-ready).
    - Uses 'instructions' for system guidance, 'input' for user content.
    - Controls: reasoning.effort, text.verbosity, max_output_tokens.
    - Avoids temperature/top_p (not needed for GPT-5).
    """
    client = get_client()
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "reasoning": {"effort": effort},
        "max_output_tokens": max_tokens,
    }
    if verbosity:
        kwargs["text"] = {"verbosity": verbosity}

    resp = client.responses.create(**kwargs)
    return (resp.output_text or "").strip()

# ---------------- Agents with tuned params ----------------

def planner_agent(idea: str) -> str:
    instr = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
             "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    prompt = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(instr, prompt, effort="minimal", verbosity="low", max_tokens=450)

def market_analysis_agent(idea: str) -> str:
    instr = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
             "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    prompt = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=1000)

def competition_analysis_agent(idea: str) -> str:
    instr = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
             "summarize opportunities to differentiate, and note any moats.")
    prompt = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=950)

def financial_feasibility_agent(idea: str) -> str:
    instr = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
             "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic explicitly.")
    prompt = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=1050)

def gtm_strategy_agent(idea: str) -> str:
    instr = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
             "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    prompt = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=850)

def risks_analysis_agent(idea: str) -> str:
    instr = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
             "likelihood/impact and brief mitigations.")
    prompt = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=800)

def critic_agent(idea: str, analyses: Dict[str, str]) -> str:
    instr = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
             "Return a bullet list of fixes and questions to validate.")
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    prompt = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=850)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> str:
    instr = ("You are a precise management consultant. Synthesize into:\n"
             "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
             "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
             "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
             "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    prompt = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
              "Produce the final consolidated report in Markdown.")
    return _respond(instr, prompt, effort="high", verbosity="medium", max_tokens=1500)
