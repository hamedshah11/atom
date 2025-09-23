import os
from typing import Dict, Optional
from openai import OpenAI

# -------- Model selection --------
# Default to the same model everywhere (Planner works → reuse that)
MODEL_PLANNER = os.getenv("PLANNER_MODEL", "gpt-4.1")
MODEL_ANALYST = os.getenv("ANALYST_MODEL", MODEL_PLANNER)   # reuse planner's model by default
MODEL_CRITIC  = os.getenv("CRITIC_MODEL",  MODEL_PLANNER)
MODEL_SYNTH   = os.getenv("SYNTH_MODEL",   MODEL_PLANNER)

# -------- Lazy singleton client --------
_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    """Create a single OpenAI client using OPENAI_API_KEY env (set by Streamlit secrets)."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Allow runtime assignment by code elsewhere (e.g., set later in UI)
            # OpenAI() can also read from env at call-time, but we keep it explicit.
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _chat(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Minimal, safe wrapper for Chat Completions (no temperature/custom params).
    This avoids 'unsupported parameter' errors on some models.
    """
    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    return (resp.choices[0].message.content or "").strip()

# -------- Agents --------

def planner_agent(idea: str) -> str:
    sys = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
           "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    usr = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _chat(MODEL_PLANNER, sys, usr)

def market_analysis_agent(idea: str) -> str:
    sys = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
           "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    usr = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    return _chat(MODEL_ANALYST, sys, usr)

def competition_analysis_agent(idea: str) -> str:
    sys = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
           "and summarize opportunities to differentiate.")
    usr = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _chat(MODEL_ANALYST, sys, usr)

def financial_feasibility_agent(idea: str) -> str:
    sys = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
           "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic.")
    usr = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _chat(MODEL_ANALYST, sys, usr)

def gtm_strategy_agent(idea: str) -> str:
    sys = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
           "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    usr = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _chat(MODEL_ANALYST, sys, usr)

def risks_analysis_agent(idea: str) -> str:
    sys = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
           "likelihood/impact and brief mitigations.")
    usr = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _chat(MODEL_ANALYST, sys, usr)

def critic_agent(idea: str, analyses: Dict[str, str]) -> str:
    sys = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
           "Return a bullet list of fixes and questions to validate.")
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    usr = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _chat(MODEL_CRITIC, sys, usr)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> str:
    sys = ("You are a precise management consultant. Synthesize into:\n"
           "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
           "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
           "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
           "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    usr = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
           "Produce the final consolidated report in Markdown.")
    return _chat(MODEL_SYNTH, sys, usr)
