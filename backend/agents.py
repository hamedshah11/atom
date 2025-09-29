import os
import json
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI

# ===== Model & behavior knobs =====
# Use gpt-3.5-turbo as default - it's reliable and cheaper
# Note: There is NO gpt-5-mini model yet!
MODEL_ALL = os.getenv("MODEL_ALL", "gpt-3.5-turbo")
VERBOSITY_ALL = os.getenv("VERBOSITY", "low")
USE_SERPER = os.getenv("USE_SERPER", "0") == "1"
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

# Rate limiting - crucial for avoiding errors
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "2.0"))

_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
    return _client

def _debug_log(message: str):
    """Print debug messages if debug mode is on"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def _chat_completion(**kwargs) -> Any:
    """Use OpenAI chat completions API with rate limiting"""
    # IMPORTANT: Add delay to prevent rate limit errors
    time.sleep(RATE_LIMIT_DELAY)
    
    try:
        response = get_client().chat.completions.create(**kwargs)
        return response
    except Exception as e:
        _debug_log(f"API Error: {type(e).__name__}: {str(e)}")
        raise

def _safe_output_text(resp: Any) -> str:
    """Extract text from OpenAI chat completion response"""
    if resp and hasattr(resp, 'choices') and resp.choices:
        if hasattr(resp.choices[0], 'message') and hasattr(resp.choices[0].message, 'content'):
            content = resp.choices[0].message.content
            return content.strip() if content else ""
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
    Simple, robust OpenAI Chat API helper
    """
    # Build system prompt
    system_content = instructions
    if verbosity == "low":
        system_content += "\n\nBe concise and direct."
    
    # Build messages
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    
    # Temperature based on effort
    temperature_map = {"minimal": 0.3, "low": 0.5, "medium": 0.7, "high": 0.9}
    temperature = temperature_map.get(effort, 0.7)
    
    try:
        response = _chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        text = _safe_output_text(response)
        
        if text:
            return text
        else:
            return "⚠️ No response generated. Please try again."
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return "⚠️ Rate limit hit. Please wait a moment and try again with fewer requests."
        elif "model" in error_msg.lower():
            return f"⚠️ Model '{model}' not available. Please use gpt-3.5-turbo or gpt-4-turbo."
        else:
            return f"⚠️ API Error: {error_msg[:200]}"

# ====================== Serper helpers ======================
def _format_serper_block(results: list[dict], label: str = "Market Research") -> str:
    if not results:
        return ""
    lines = [f"\n{label}:"]
    for i, it in enumerate(results[:3], 1):  # Limit to 3 results
        title = it.get("title", "")[:100]
        snippet = it.get("snippet", "")[:200]
        lines.append(f"{i}. {title}\n   {snippet}")
    return "\n".join(lines)

# ======================= Agents =======================

def planner_agent(idea: str) -> str:
    instr = (
        "You are a planning agent. Create 4-6 concise steps to analyze this business idea. "
        "Focus on: market size, competition, financials, go-to-market, and risks."
    )
    prompt = f"Business Idea: {idea}\n\nCreate a brief analysis plan."
    return _respond(instr, prompt, effort="minimal", max_tokens=400)

def market_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} {idea.split()[0]} market size statistics"
            results = web_search_serper(q, num=3)
            serper_block = _format_serper_block(results)
        except:
            pass

    instr = (
        "You are a market analyst. Estimate TAM, SAM, and SOM for this business. "
        "Format: 1) Brief assumptions (2 sentences), 2) Simple table, 3) Final line: TAM: X, SAM: Y, SOM: Z"
    )
    prompt = f"Business: {idea}\nRegion: {region}{serper_block}\n\nProvide market analysis."
    return _respond(instr, prompt, max_tokens=800)

def competition_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} {idea.split()[0]} competitors companies"
            serper_block = _format_serper_block(web_search_serper(q, num=3))
        except:
            pass
    
    instr = "You are a competition analyst. List 3-5 key competitors with brief positioning and differentiation opportunities."
    prompt = f"Business: {idea}\nRegion: {region}{serper_block}\n\nAnalyze competition."
    return _respond(instr, prompt, max_tokens=700)

def financial_feasibility_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a finance analyst. Outline: revenue model, pricing, key costs, margins, "
        "and simple 3-year projection. Use the local currency."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nProvide financial analysis."
    return _respond(instr, prompt, max_tokens=800)

def gtm_strategy_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a GTM strategist. Define: target customers, marketing channels, "
        "key messages, and 90-day launch plan with 5-7 action items."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nCreate GTM strategy."
    return _respond(instr, prompt, max_tokens=700)

def risks_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    instr = "You are a risk analyst. List 5 major risks (regulatory, market, operational, financial, competitive) with impact level and mitigation."
    prompt = f"Business: {idea}\nRegion: {region}\n\nIdentify key risks."
    return _respond(instr, prompt, max_tokens=600)

def critic_agent(idea: str, analyses: Dict[str, str], region: str = "Pakistan") -> str:
    instr = "You are a business critic. Review the analyses and identify 3-5 major gaps, contradictions, or concerns."
    analyses_summary = "\n".join([f"{k}: {v[:200]}..." for k, v in analyses.items()])
    prompt = f"Business: {idea}\nRegion: {region}\n\nAnalyses:\n{analyses_summary}\n\nProvide critical review."
    return _respond(instr, prompt, max_tokens=600)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a management consultant. Create a final report with:\n"
        "1. Executive Summary (3-4 bullets with Go/No-Go recommendation)\n"
        "2. Key Insights (top 5 findings)\n"
        "3. Next Steps (3-5 actions)\n"
        "Use markdown formatting."
    )
    # Keep it concise to avoid token limits
    analyses_summary = "\n".join([f"{k}: {v[:150]}..." for k, v in analyses.items()])
    prompt = (
        f"Business: {idea}\nRegion: {region}\n\n"
        f"Key findings:\n{analyses_summary}\n\n"
        f"Critique: {critique[:200]}...\n\n"
        "Create final report."
    )
    return _respond(instr, prompt, effort="high", max_tokens=1200)
