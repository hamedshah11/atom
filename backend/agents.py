import os
from typing import Optional, Dict, Any
from openai import OpenAI

# ===== Model & behavior knobs (can be set via Streamlit secrets) =====
MODEL_ALL      = os.getenv("MODEL_ALL",      os.getenv("model_all", "gpt-5"))
VERBOSITY_ALL  = os.getenv("VERBOSITY",      os.getenv("verbosity", "low"))
ENABLE_WEB     = os.getenv("ENABLE_WEB_SEARCH", "0") == "1"   # optional Responses web_search tool

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _responses_create(**kwargs) -> Any:
    return get_client().responses.create(**kwargs)

def _respond(
    instructions: str,
    prompt: str,
    *,
    model: str = MODEL_ALL,
    effort: Optional[str] = "medium",
    verbosity: Optional[str] = VERBOSITY_ALL,
    max_tokens: int = 1000,
    use_web_search: bool = False
) -> str:
    """
    Responses API wrapper with:
      - instructions + input
      - reasoning.effort
      - text.verbosity (for GPT-5)
      - optional web_search tool (if ENABLE_WEB_SEARCH=1 and your key supports it)
      - stateless (store=False)
    """
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "store": False,
        "max_output_tokens": max_tokens,
    }
    if effort:
        payload["reasoning"] = {"effort": effort}
    if verbosity:
        payload["text"] = {"verbosity": verbosity}

    # Try with web_search tool if requested (and fall back if unsupported)
    if use_web_search:
        try:
            with_tool = dict(payload)
            with_tool["tools"] = [{"type": "web_search"}]  # or web_search_preview
            r = _responses_create(**with_tool)
            return (r.output_text or "").strip()
        except Exception:
            pass  # fall back to plain call

    r = _responses_create(**payload)
    return (r.output_text or "").strip()

# ======================= Agents (all return str) =======================

def planner_agent(idea: str) -> str:
    instr = (
        "You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
        "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets."
    )
    prompt = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(instr, prompt, effort="minimal", verbosity="low", max_tokens=420)

def market_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    """
    Region-aware, compact, and structured.
    Guarantees a final single line: TAM: <X>, SAM: <Y>, SOM: <Z> (with units).
    """
    instr = (
        "You are a market sizing analyst. Using the target region and reasonable public assumptions, "
        "estimate TAM, SAM, and SOM for the proposed business. Keep output compact, with:\n"
        "1) A 2–4 sentence method & assumptions (participation rates, price, capacity, etc.)\n"
        "2) A 3-row Markdown table with columns: Segment | Units | Value\n"
        "3) A FINAL single line strictly formatted as: TAM: <X>, SAM: <Y>, SOM: <Z> (include units)\n"
        "If the region is Pakistan, default currency to PKR; if you must convert, state the FX briefly in the method."
    )
    prompt = (
        f"Business Idea:\n{idea}\n\n"
        f"Target Region: {region}\n\n"
        "Provide a brief market overview and compute TAM/SAM/SOM as instructed."
    )
    # Give a bit more budget but keep verbosity low so it stays compact.
    return _respond(
        instr, prompt,
        effort="medium", verbosity="low", max_tokens=1200,
        use_web_search=ENABLE_WEB
    )

def competition_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a competition analyst. Identify 3–6 key competitors/alternatives in the target region, "
        "compare positioning briefly, and summarize opportunities to differentiate."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nAnalyze the competitive landscape."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=900)

def financial_feasibility_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a finance analyst. Outline revenue model, price points, COGS, gross margin, opex buckets, "
        "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic. Keep it compact."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nEvaluate financial feasibility."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=950)

def gtm_strategy_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
        "Add a simple funnel (impressions→leads→conversions) with baseline assumptions."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nPropose GTM strategy."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=750)

def risks_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
        "likelihood/impact and brief mitigations."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nIdentify key risks & mitigations."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=700)

def critic_agent(idea: str, analyses: Dict[str, str], region: str = "Pakistan") -> str:
    instr = (
        "You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
        "Return a bullet list of fixes and the top questions to validate in the specified region."
    )
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=800)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a precise management consultant. Synthesize into:\n"
        "• Executive Summary (3–6 bullets; explicit go/no-go)\n"
        "• LEAN CANVAS (Problem, Segments, UVP, Solution, Channels, Revenue, Costs, Metrics, Unfair Advantage)\n"
        "• MARKET (TAM/SAM/SOM + method; competitor summary)\n"
        "• FEASIBILITY (unit econ, breakeven, key risks, top 3 unknowns)\n"
        "Keep concise, region-aware, and include currency units."
    )
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    prompt = (
        f"Business Idea:\n{idea}\nTarget Region: {region}\n\n"
        f"Inputs:\n{blob}\n\nCritique:\n{critique}\n\n"
        "Produce the final consolidated report in Markdown. Start with the Executive Summary."
    )
    return _respond(instr, prompt, effort="high", verbosity="medium", max_tokens=1400)
