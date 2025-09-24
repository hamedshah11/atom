import os
from typing import Optional, Dict, Any
from openai import OpenAI

# Primary model and safe fallbacks
MODEL_ALL = os.getenv("MODEL_ALL", "gpt-5")
FALLBACKS = ["gpt-5-mini", "gpt-4.1"]  # only used on model/access/quota errors

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _responses_call(model: str, instructions: str, prompt: str,
                    effort: Optional[str], verbosity: Optional[str],
                    max_tokens: int, store: bool = False) -> str:
    """
    Adaptive call:
      1) Try full shape: instructions, input, reasoning.effort, text.verbosity, max_output_tokens.
      2) If API rejects fields ('reasoning', 'verbosity', 'max_output_tokens'), strip & retry.
      3) If output budget too large, shrink and retry.
      4) As a last shape fallback, try minimal payload: model + input only.
    Returns response.output_text.
    """
    client = get_client()

    # Attempt order with decreasing feature set
    attempts: list[Dict[str, Any]] = []

    # A) Full featured
    full: Dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    if effort:
        full["reasoning"] = {"effort": effort}
    if verbosity:
        full["text"] = {"verbosity": verbosity}
    attempts.append(full)

    # B) No text.verbosity
    if "text" in full:
        no_text = dict(full)
        no_text.pop("text", None)
        attempts.append(no_text)

    # C) No reasoning
    if "reasoning" in full:
        no_reason = dict(full)
        no_reason.pop("reasoning", None)
        attempts.append(no_reason)

    # D) No reasoning, no text
    if "reasoning" in full or "text" in full:
        bare_controls = {k: v for k, v in full.items() if k not in ("reasoning", "text")}
        attempts.append(bare_controls)

    # E) Minimal (drop max_output_tokens too)
    minimal = {"model": model, "input": [{"role": "user", "content": prompt}], "store": store}
    # keep instructions separately if needed
    if instructions:
        minimal["instructions"] = instructions
    attempts.append(minimal)

    last_err: Optional[Exception] = None
    # For shrinking output budget when API complains
    budgets = [max_tokens, max(600, int(max_tokens * 0.8)), 500, 350]

    for payload in attempts:
        for budget in budgets:
            payload_try = dict(payload)
            if "max_output_tokens" in payload_try:
                payload_try["max_output_tokens"] = budget
            try:
                resp = client.responses.create(**payload_try)
                return (resp.output_text or "").strip()
            except Exception as e:
                msg = str(e).lower()
                last_err = e
                # If obviously not related to params/budget, bail early to let outer logic handle model fallback
                # We still continue loop to try more shapes/budgets, unless it's clearly unrecoverable.
                # Common recoverable fragments:
                recoverable = any(k in msg for k in [
                    "unsupported", "unrecognized", "invalid request", "max_output_tokens",
                    "verbosity", "reasoning", "too_large", "exceeds", "parameter"
                ])
                if not recoverable:
                    # let outer fallback logic decide (model/access/quota)
                    break
        # proceed to next shape

    # If we tried all shapes/budgets and still failed with parameter-ish errors, raise
    if last_err:
        raise last_err
    raise RuntimeError("Unknown error calling Responses API")

def _respond(instructions: str, prompt: str, *,
             effort: Optional[str] = "medium",
             verbosity: Optional[str] = None,
             max_tokens: int = 1000) -> str:
    """
    Try primary model with adaptive shapes/budgets; only fall back to alternates
    on model/access/quota errors. Returns plain text.
    """
    models_to_try = [MODEL_ALL] + [m for m in FALLBACKS if m != MODEL_ALL]
    last_err: Optional[Exception] = None

    for m in models_to_try:
        try:
            return _responses_call(
                model=m,
                instructions=instructions,
                prompt=prompt,
                effort=effort,
                verbosity=verbosity,
                max_tokens=max_tokens,
                store=False,  # stateless for Streamlit Cloud
            )
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            model_issue = any(k in msg for k in [
                "model not found", "does not exist", "unknown model",
                "insufficient_quota", "access", "not available", "rate limit"
            ])
            if model_issue:
                # try next model
                continue
            # Non-model issue (params, formatting) already exhausted adaptive shapes: bubble up
            raise

    # All models failed for model/access/quota reasons
    raise RuntimeError(f"OpenAI Responses failed for all models: {last_err}")

# ---------------- Agents (all return str) ----------------

def planner_agent(idea: str) -> str:
    instr = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
             "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    prompt = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(instr, prompt, effort="minimal", verbosity="low", max_tokens=420)

def market_analysis_agent(idea: str) -> str:
    instr = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
             "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    prompt = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=900)

def competition_analysis_agent(idea: str) -> str:
    instr = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
             "summarize opportunities to differentiate, and note any moats.")
    prompt = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _respond(instr, prompt, effort="medium", verbosity="medium", max_tokens=850)

def financial_feasibility_agent(idea: str) -> str:
    instr = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
             "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic explicitly.")
    prompt = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=900)

def gtm_strategy_agent(idea: str) -> str:
    instr = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
             "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    prompt = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=750)

def risks_analysis_agent(idea: str) -> str:
    instr = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
             "likelihood/impact and brief mitigations.")
    prompt = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _respond(instr, prompt, effort="low", verbosity="low", max_tokens=700)

def critic_agent(idea: str, analyses: Dict[str, str]) -> str:
    instr = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
             "Return a bullet list of fixes and questions to validate.")
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    prompt = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=800)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> str:
    instr = ("You are a precise management consultant. Synthesize into:\n"
             "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
             "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
             "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
             "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    prompt = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
              "Produce the final consolidated report in Markdown.")
    return _respond(instr, prompt, effort="high", verbosity="medium", max_tokens=1200)
