import os
from typing import Dict, Optional
from openai import OpenAI

# --- Model selection (default everything to o4-mini; override via env if needed) ---
MODEL_ALL = os.getenv("MODEL_ALL", "o4-mini")       # e.g. "gpt-5-mini" for GPT-5, or "gpt-4.1" etc.
VERBOSITY_ALL = os.getenv("VERBOSITY", "medium")   # new parameter for GPT-5 models

# --- Singleton client using OPENAI_API_KEY (set by Streamlit secrets) ---
_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _respond(model: str, system_prompt: str, user_prompt: str,
             effort: str = "medium", max_tokens: int = 1200) -> str:
    """
    Calls the OpenAI Responses API for the given model.
    Supports new GPT-5/4.1 parameters (e.g. verbosity) when needed.
    """
    client = get_client()
    # Build request parameters
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        "reasoning": {"effort": effort},
        "max_output_tokens": max_tokens,
    }
    # If using a GPT-5 model, include verbosity control
    if model.startswith("gpt-5"):
        payload["text"] = {"verbosity": VERBOSITY_ALL}
    resp = client.responses.create(**payload)
    # Return the final output text (or empty string)
    return (resp.output_text or "").strip()

# ---------------- AGENTS ----------------

def planner_agent(idea: str) -> str:
    sys = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
           "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    usr = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(MODEL_ALL, sys, usr, effort="low", max_tokens=700)

def market_analysis_agent(idea: str) -> str:
    sys = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
           "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    usr = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    return _respond(MODEL_ALL, sys, usr, effort="medium", max_tokens=1100)

def competition_analysis_agent(idea: str) -> str:
    sys = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
           "and summarize opportunities to differentiate.")
    usr = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _respond(MODEL_ALL, sys, usr, effort="medium", max_tokens=1000)

def financial_feasibility_agent(idea: str) -> str:
    sys = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
           "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic.")
    usr = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _respond(MODEL_ALL, sys, usr, effort="medium", max_tokens=1100)

def gtm_strategy_agent(idea: str) -> str:
    sys = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
           "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    usr = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _respond(MODEL_ALL, sys, usr, effort="low", max_tokens=900)

def risks_analysis_agent(idea: str) -> str:
    sys = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
           "likelihood/impact and brief mitigations.")
    usr = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _respond(MODEL_ALL, sys, usr, effort="low", max_tokens=900)

def critic_agent(idea: str, analyses: Dict[str, str]) -> str:
    sys = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
           "Return a bullet list of fixes and questions to validate.")
    # Prepare the analyses blob
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    usr = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _respond(MODEL_ALL, sys, usr, effort="medium", max_tokens=900)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> str:
    sys = ("You are a precise management consultant. Synthesize into:\n"
           "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
           "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
           "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
           "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    usr = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
           "Produce the final consolidated report in Markdown.")
    return _respond(MODEL_ALL, sys, usr, effort="medium", max_tokens=1400)
