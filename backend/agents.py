import os
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ===== Model & behavior knobs (override via Streamlit secrets) =====
MODEL_ALL      = os.getenv("MODEL_ALL",      os.getenv("model_all", "gpt-5"))
VERBOSITY_ALL  = os.getenv("VERBOSITY",      os.getenv("verbosity", "low"))

# Serper toggle (cheaper search)
USE_SERPER     = os.getenv("USE_SERPER", "0") == "1"

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _responses(**kwargs) -> Any:
    return get_client().responses.create(**kwargs)

def _not_empty(text: Optional[str]) -> bool:
    return bool(text and text.strip())

def _safe_output_text(resp: Any) -> str:
    return (resp.output_text or "").strip()

def _try_payloads(payloads: List[Dict[str, Any]]) -> str:
    """
    Try a list of Responses payloads until one returns non-empty text.
    If all succeed but return empty, raise a RuntimeError so callers surface an error instead of blank UI.
    """
    last_exc: Optional[Exception] = None
    for p in payloads:
        try:
            r = _responses(**p)
            txt = _safe_output_text(r)
            if _not_empty(txt):
                return txt
        except Exception as e:
            last_exc = e
            # keep trying next shape
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("LLM returned empty output for all attempts")

def _respond(
    instructions: str,
    prompt: str,
    *,
    model: str = MODEL_ALL,
    effort: Optional[str] = "medium",
    verbosity: Optional[str] = VERBOSITY_ALL,
    max_tokens: int = 1000,
) -> str:
    """
    Robust Responses API helper:
      • Tries several request shapes (instructions+string, messages array, with/without reasoning/verbosity).
      • Adjusts token budget & verbosity if needed.
      • Never returns empty; throws if no variant produced text.
    """
    # Common base
    base: Dict[str, Any] = {
        "model": model,
        "store": False,
    }

    shapes: List[Dict[str, Any]] = []

    # A) instructions + input (string), full controls
    sA = dict(base)
    sA["instructions"] = instructions
    sA["input"] = prompt
    sA["max_output_tokens"] = max_tokens
    if effort:
        sA["reasoning"] = {"effort": effort}
    if verbosity:
        sA["text"] = {"verbosity": verbosity}
    shapes.append(sA)

    # B) messages array (system + user), same controls
    sB = dict(base)
    sB["input"] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]
    sB["max_output_tokens"] = max_tokens
    if effort:
        sB["reasoning"] = {"effort": effort}
    if verbosity:
        sB["text"] = {"verbosity": verbosity}
    shapes.append(sB)

    # C) instructions + input, drop verbosity
    sC = dict(sA)
    sC.pop("text", None)
    shapes.append(sC)

    # D) messages array, drop verbosity
    sD = dict(sB)
    sD.pop("text", None)
    shapes.append(sD)

    # E) instructions + input, drop reasoning too
    sE = dict(sC)
    sE.pop("reasoning", None)
    shapes.append(sE)

    # F) messages array, drop reasoning too
    sF = dict(sD)
    sF.pop("reasoning", None)
    shapes.append(sF)

    # G) bump tokens a bit and verbosity → medium, messages array
    sG = dict(sB)
    sG["max_output_tokens"] = max(max_tokens, 1200)
    sG["text"] = {"verbosity": "medium"}
    shapes.append(sG)

    return _try_payloads(shapes)

# ====================== Optional: Serper helpers ======================
def _format_serper_block(results: list[dict], label: str = "External Signals") -> str:
    if not results:
        return ""
    lines = [f"{label} (top results):"]
    for i, it in enumerate(results, 1):
        title = it.get("title") or ""
        link = it.get("link") or ""
        snippet = it.get("snippet") or ""
        lines.append(f"{i}. {title}\n   {snippet}\n   {link}")
    return "\n".join(lines)

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
    Region-aware, compact, structured. Uses Serper (if enabled) to ground assumptions.
    Guarantees a final single line: TAM: <X>, SAM: <Y>, SOM: <Z> (with units).
    Never returns empty.
    """
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} padel market size participation rate average court price"
            results = web_search_serper(q, num=5)
            serper_block = _format_serper_block(results)
        except Exception:
            serper_block = ""

    instr = (
        "You are a market sizing analyst. Using the target region and reasonable public assumptions, "
        "estimate TAM, SAM, and SOM for the proposed business. Keep output compact, with:\n"
        "1) A 2–4 sentence method & assumptions (participation rates, price, capacity, etc.). If you used external signals, mention them.\n"
        "2) A 3-row Markdown table with columns: Segment | Units | Value\n"
        "3) A FINAL single line strictly formatted as: TAM: <X>, SAM: <Y>, SOM: <Z> (include units)\n"
        "If the region is Pakistan, default currency to PKR; if converting, state FX briefly in the method."
    )
    prompt = (
        f"Business Idea:\n{idea}\n\n"
        f"Target Region: {region}\n\n"
        f"{serper_block}\n\n"
        "Provide a concise market overview and compute TAM/SAM/SOM exactly as instructed."
    )
    txt = _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=1200)
    if not _not_empty(txt):
        # Provide a clear placeholder—UI won’t look “blank”
        return "⚠️ Market Analysis: no content generated."
    return txt

def competition_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} padel clubs competitors booking platforms"
            serper_block = _format_serper_block(web_search_serper(q, num=5))
        except Exception:
            serper_block = ""
    instr = (
        "You are a competition analyst. Identify 3–6 key competitors/alternatives in the target region, "
        "compare positioning briefly, and summarize opportunities to differentiate."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\n{serper_block}\n\nAnalyze the competitive landscape."
    txt = _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=900)
    if not _not_empty(txt):
        return "⚠️ Competition: no content generated."
    return txt

def financial_feasibility_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a finance analyst. Outline revenue model, price points, COGS, gross margin, opex buckets, "
        "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic. Keep it compact."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nEvaluate financial feasibility."
    txt = _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=950)
    if not _not_empty(txt):
        return "⚠️ Financial Feasibility: no content generated."
    return txt

def gtm_strategy_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
        "Add a simple funnel (impressions→leads→conversions) with baseline assumptions."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nPropose GTM strategy."
    txt = _respond(instr, prompt, effort="low", verbosity="low", max_tokens=750)
    if not _not_empty(txt):
        return "⚠️ GTM: no content generated."
    return txt

def risks_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
        "likelihood/impact and brief mitigations."
    )
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nIdentify key risks & mitigations."
    txt = _respond(instr, prompt, effort="low", verbosity="low", max_tokens=700)
    if not _not_empty(txt):
        return "⚠️ Risks: no content generated."
    return txt

def critic_agent(idea: str, analyses: Dict[str, str], region: str = "Pakistan") -> str:
    instr = (
        "You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
        "Return a bullet list of fixes and the top questions to validate in the specified region."
    )
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    prompt = f"Business Idea:\n{idea}\nTarget Region: {region}\n\nAnalyses:\n{blob}\n\nProvide critique."
    txt = _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=800)
    if not _not_empty(txt):
        return "⚠️ Critique: no content generated."
    return txt

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
    txt = _respond(instr, prompt, effort="high", verbosity="medium", max_tokens=1400)
    if not _not_empty(txt):
        return "⚠️ Final Synthesis: no content generated."
    return txt
