import os
import json
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ===== Model & behavior knobs (override via Streamlit secrets) =====
MODEL_ALL      = os.getenv("MODEL_ALL",      os.getenv("model_all", "gpt-3.5-turbo"))
VERBOSITY_ALL  = os.getenv("VERBOSITY",      os.getenv("verbosity", "low"))

# Serper toggle (cheaper search)
USE_SERPER     = os.getenv("USE_SERPER", "0") == "1"

# Debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "1") == "1"

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client

def _debug_log(message: str):
    """Print debug messages if debug mode is on"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def _chat_completion(**kwargs) -> Any:
    """Use the actual OpenAI chat completions API"""
    _debug_log(f"Making API call with model: {kwargs.get('model', 'default')}")
    _debug_log(f"Messages: {json.dumps(kwargs.get('messages', []), indent=2)[:500]}...")
    
    try:
        response = get_client().chat.completions.create(**kwargs)
        _debug_log(f"API Response received. Choices: {len(response.choices)}")
        return response
    except Exception as e:
        _debug_log(f"API Error: {type(e).__name__}: {str(e)}")
        raise

def _not_empty(text: Optional[str]) -> bool:
    return bool(text and text.strip())

def _safe_output_text(resp: Any) -> str:
    """Extract text from OpenAI chat completion response with debugging"""
    if resp is None:
        _debug_log("Response is None")
        return ""
    
    if hasattr(resp, 'choices') and resp.choices:
        choice = resp.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            content = choice.message.content
            _debug_log(f"Extracted content: {content[:100]}..." if content else "Content is None/empty")
            return content.strip() if content else ""
        else:
            _debug_log("No message.content in choice")
    else:
        _debug_log("No choices in response")
    
    return ""

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
    Robust OpenAI Chat API helper with detailed debugging
    """
    _debug_log(f"\n=== Starting _respond ===")
    _debug_log(f"Model: {model}, Max tokens: {max_tokens}")
    _debug_log(f"Instructions length: {len(instructions)}")
    _debug_log(f"Prompt length: {len(prompt)}")
    
    # Build system prompt with verbosity guidance
    system_content = instructions
    if verbosity == "low":
        system_content += "\n\nBe concise and to the point."
    elif verbosity == "medium":
        system_content += "\n\nProvide moderate detail."
    elif verbosity == "high":
        system_content += "\n\nProvide comprehensive detail."
    
    # Build the messages for chat completion
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # Temperature based on effort
    temperature_map = {"minimal": 0.3, "low": 0.5, "medium": 0.7, "high": 0.9}
    temperature = temperature_map.get(effort, 0.7)
    
    try:
        _debug_log(f"Calling OpenAI API...")
        response = _chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        text = _safe_output_text(response)
        
        if not _not_empty(text):
            _debug_log("WARNING: Received empty response from API")
            # Try with a simpler prompt
            _debug_log("Retrying with simplified prompt...")
            messages = [
                {"role": "user", "content": f"{instructions}\n\n{prompt[:500]}"}  # Truncate if too long
            ]
            response = _chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.5
            )
            text = _safe_output_text(response)
        
        if not _not_empty(text):
            # Last attempt - very simple
            _debug_log("Final attempt with minimal prompt...")
            response = _chat_completion(
                model=model,
                messages=[{"role": "user", "content": "Please respond with any text to confirm the API is working."}],
                max_tokens=50,
                temperature=0
            )
            test_text = _safe_output_text(response)
            if _not_empty(test_text):
                _debug_log(f"API is working, but original prompt may be problematic. Test response: {test_text}")
                return "⚠️ API is working but couldn't generate content for this specific request. Try simplifying the input."
            else:
                _debug_log("API is not returning any content at all")
                raise RuntimeError("API is not returning any content")
        
        _debug_log(f"Success! Returning {len(text)} characters")
        return text
        
    except Exception as e:
        _debug_log(f"Exception in _respond: {type(e).__name__}: {str(e)}")
        raise

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
    _debug_log("\n>>> PLANNER AGENT CALLED")
    instr = (
        "You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
        "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets."
    )
    prompt = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    return _respond(instr, prompt, effort="minimal", verbosity="low", max_tokens=420)

def market_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    _debug_log("\n>>> MARKET ANALYSIS AGENT CALLED")
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} market size trends statistics"
            results = web_search_serper(q, num=5)
            serper_block = _format_serper_block(results)
        except Exception:
            serper_block = ""

    instr = (
        "You are a market sizing analyst. Using the target region and reasonable public assumptions, "
        "estimate TAM, SAM, and SOM for the proposed business. Keep output compact, with:\n"
        "1) A 2–4 sentence method & assumptions (participation rates, price, capacity, etc.).\n"
        "2) A 3-row Markdown table with columns: Segment | Units | Value\n"
        "3) A FINAL single line strictly formatted as: TAM: <X>, SAM: <Y>, SOM: <Z> (include units)\n"
    )
    prompt = (
        f"Business Idea:\n{idea}\n\n"
        f"Target Region: {region}\n\n"
        f"{serper_block}\n\n"
        "Provide a concise market overview and compute TAM/SAM/SOM exactly as instructed."
    )
    txt = _respond(instr, prompt, effort="medium", verbosity="low", max_tokens=1200)
    if not _not_empty(txt):
        return "⚠️ Market Analysis: no content generated."
    return txt

def competition_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    _debug_log("\n>>> COMPETITION ANALYSIS AGENT CALLED")
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} competitors market players industry analysis"
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
    _debug_log("\n>>> FINANCIAL FEASIBILITY AGENT CALLED")
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
    _debug_log("\n>>> GTM STRATEGY AGENT CALLED")
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
    _debug_log("\n>>> RISKS ANALYSIS AGENT CALLED")
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
    _debug_log("\n>>> CRITIC AGENT CALLED")
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
    _debug_log("\n>>> SYNTHESIZER AGENT CALLED")
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
