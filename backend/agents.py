import os
from typing import Dict, Optional, Tuple
from openai import OpenAI

# Default model for all agents (override in secrets or env if needed)
PRIMARY_MODEL = os.getenv("MODEL_ALL", "gpt-5")
FALLBACKS = ["gpt-5-mini", "gpt-4.1"]  # used only if primary throws model/access errors

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _call_responses(model: str, instructions: str, prompt: str,
                    effort: str, verbosity: Optional[str], max_tokens: int):
    client = get_client()
    kwargs = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "reasoning": {"effort": effort},   # minimal | low | medium | high
        "max_output_tokens": max_tokens,
        "store": False,                    # stateless for Streamlit Cloud
    }
    if verbosity:
        kwargs["text"] = {"verbosity": verbosity}  # low | medium | high
    return client.responses.create(**kwargs)

def _respond(instructions: str, prompt: str, *,
             effort: str = "medium", verbosity: Optional[str] = None,
             max_tokens: int = 1200) -> Tuple[str, str]:
    """
    Returns (text, model_used). Attempts PRIMARY_MODEL, then fallbacks on specific API errors.
    """
    tried = []
    models_to_try = [PRIMARY_MODEL] + [m for m in FALLBACKS if m != PRIMARY_MODEL]

    last_err = None
    for m in models_to_try:
        try:
            resp = _call_responses(m, instructions, prompt, effort, verbosity, max_tokens)
            return (resp.output_text.strip() if resp.output_text else "", m)
        except Exception as e:
            last_err = e
            tried.append(m)
            # Failover only on common model/access errors; otherwise re-raise quickly
            msg = str(e).lower()
            recoverable = any(k in msg for k in [
                "model not found", "does not exist", "unknown model",
                "insufficient_quota", "access", "unsupported", "not available"
            ])
            if not recoverable and m == models_to_try[0]:
                # if it's not a typical model/access error on the primary, re-raise
                raise
            # else: try next model
            continue
    # If we get here, all attempts failed
    raise RuntimeError(f"OpenAI Responses call failed for models {tried}: {last_err}")

# ---------------- Agents with tuned params (GPT-5-friendly) ----------------

def planner_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
             "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    prompt = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(instr, prompt, effort="minimal", verbosity="low", max_tokens=450)

def market_analysis_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
             "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    prompt = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=1000)

def competition_analysis_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
             "summarize opportunities to differentiate, and note any moats.")
    prompt = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=950)

def financial_feasibility_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
             "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic explicitly.")
    prompt = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=1050)

def gtm_strategy_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
             "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    prompt = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=850)

def risks_analysis_agent(idea: str) -> Tuple[str, str]:
    instr = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
             "likelihood/impact and brief mitigations.")
    prompt = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=800)

def critic_agent(idea: str, analyses: Dict[str, str]) -> Tuple[str, str]:
    instr = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
             "Return a bullet list of fixes and questions to validate.")
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    prompt = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=850)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> Tuple[str, str]:
    instr = ("You are a precise management consultant. Synthesize into:\n"
             "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
             "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
             "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
             "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    prompt = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
              "Produce the final consolidated report in Markdown.")
    return _respond(instr, prompt, effort="high", verbosity="medium", max_tokens=1500)
