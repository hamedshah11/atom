import os
import json
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from langsmith import traceable

# ===== Model Configuration =====
# O3-mini only configuration (with gpt-4o as fallback)
AVAILABLE_MODELS = ["o3-mini", "gpt-4o"]  # o3-mini primary, gpt-4o fallback
DEFAULT_MODEL = "o3-mini"

VERBOSITY_ALL = os.getenv("VERBOSITY", "low")
USE_SERPER = os.getenv("USE_SERPER", "0") == "1"
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

# Rate limiting
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))

# LangSmith configuration
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

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

def _get_current_model() -> str:
    """Get the current model from environment - DYNAMICALLY"""
    model = os.getenv("MODEL_ALL", DEFAULT_MODEL)
    # Validate model is in available list
    if model not in AVAILABLE_MODELS:
        print(f"Warning: Model {model} not available. Using {DEFAULT_MODEL}")
        return DEFAULT_MODEL
    return model

def _is_o3_mini_model(model: str) -> bool:
    """Check if model is o3-mini reasoning model"""
    return model == "o3-mini"

@traceable(name="openai_chat_completion")
def _chat_completion(**kwargs) -> Any:
    """Use OpenAI chat completions API with rate limiting and LangSmith tracing"""
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

@traceable(name="llm_respond")
def _respond(
    instructions: str,
    prompt: str,
    *,
    model: Optional[str] = None,
    effort: Optional[str] = "medium",
    verbosity: Optional[str] = None,
    max_tokens: int = 1000,
) -> str:
    """
    OpenAI Chat API helper optimized for o3-mini
    Falls back to gpt-4o if o3-mini fails
    """
    # Get model dynamically if not provided
    if model is None:
        model = _get_current_model()
    
    # Ensure model is available
    if model not in AVAILABLE_MODELS:
        model = DEFAULT_MODEL
        _debug_log(f"Model not available, using {DEFAULT_MODEL}")
    
    if verbosity is None:
        verbosity = VERBOSITY_ALL
    
    _debug_log(f"Using model: {model}")
    
    # Try with primary model first
    try:
        return _call_model(instructions, prompt, model, effort, verbosity, max_tokens)
    except Exception as e:
        error_msg = str(e)
        
        # If o3-mini fails due to tier restrictions, fall back to gpt-4o
        if model == "o3-mini" and ("tier" in error_msg.lower() or "quota" in error_msg.lower() or "does not exist" in error_msg.lower()):
            _debug_log(f"O3-mini failed, falling back to gpt-4o: {error_msg[:100]}")
            try:
                return _call_model(instructions, prompt, "gpt-4o", effort, verbosity, max_tokens)
            except Exception as fallback_error:
                return f"⚠️ Both o3-mini and gpt-4o failed. Error: {str(fallback_error)[:200]}"
        else:
            # For other errors or if already using gpt-4o, return error message
            if "rate limit" in error_msg.lower():
                return "⚠️ Rate limit hit. Please wait a moment and try again."
            else:
                return f"⚠️ API Error: {error_msg[:200]}"

def _call_model(
    instructions: str,
    prompt: str,
    model: str,
    effort: str,
    verbosity: str,
    max_tokens: int
) -> str:
    """Internal method to call a specific model"""
    
    is_o3_mini = _is_o3_mini_model(model)
    
    # O3-mini configuration (supports streaming and tools unlike o1)
    if is_o3_mini:
        # O3-mini can use system messages (unlike o1)
        system_content = instructions
        if verbosity == "low":
            system_content += "\n\nBe concise and direct."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        api_params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,  # o3-mini uses max_completion_tokens
        }
        
        # O3-mini supports reasoning_effort
        reasoning_map = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
        api_params["reasoning_effort"] = reasoning_map.get(effort, "medium")
        
        # O3-mini does NOT support parallel_tool_calls
        # We don't use tools in this app, but if we did, we'd set:
        # api_params["parallel_tool_calls"] = False
    
    else:
        # GPT-4o configuration (fallback)
        system_content = instructions
        if verbosity == "low":
            system_content += "\n\nBe concise and direct."
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ]
        
        temperature_map = {"minimal": 0.3, "low": 0.5, "medium": 0.7, "high": 0.9}
        temperature = temperature_map.get(effort, 0.7)
        
        api_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    
    response = _chat_completion(**api_params)
    text = _safe_output_text(response)
    
    if text:
        return text
    else:
        raise Exception("No response generated")

# ====================== Serper helpers ======================
def _format_serper_block(results: list[dict], label: str = "Market Research") -> str:
    if not results:
        return ""
    lines = [f"\n{label}:"]
    for i, it in enumerate(results[:3], 1):
        title = it.get("title", "")[:100]
        snippet = it.get("snippet", "")[:200]
        lines.append(f"{i}. {title}\n   {snippet}")
    return "\n".join(lines)

# ======================= Agents =======================

@traceable(name="planner_agent")
def planner_agent(idea: str) -> str:
    instr = (
        "You are a planning agent. Create 4-6 concise steps to analyze this business idea. "
        "Focus on: market size, competition, financials, go-to-market, and risks."
    )
    prompt = f"Business Idea: {idea}\n\nCreate a brief analysis plan."
    return _respond(instr, prompt, effort="minimal", max_tokens=400)

@traceable(name="market_analysis_agent")
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

@traceable(name="competition_analysis_agent")
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

@traceable(name="financial_feasibility_agent")
def financial_feasibility_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a finance analyst. Outline: revenue model, pricing, key costs, margins, "
        "and simple 3-year projection. Use the local currency."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nProvide financial analysis."
    return _respond(instr, prompt, max_tokens=800)

@traceable(name="gtm_strategy_agent")
def gtm_strategy_agent(idea: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a GTM strategist. Define: target customers, marketing channels, "
        "key messages, and 90-day launch plan with 5-7 action items."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nCreate GTM strategy."
    return _respond(instr, prompt, max_tokens=700)

@traceable(name="risks_analysis_agent")
def risks_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    instr = "You are a risk analyst. List 5 major risks (regulatory, market, operational, financial, competitive) with impact level and mitigation."
    prompt = f"Business: {idea}\nRegion: {region}\n\nIdentify key risks."
    return _respond(instr, prompt, max_tokens=600)

@traceable(name="critic_agent")
def critic_agent(idea: str, analyses: Dict[str, str], region: str = "Pakistan") -> str:
    instr = "You are a business critic. Review the analyses and identify 3-5 major gaps, contradictions, or concerns."
    analyses_summary = "\n".join([f"{k}: {v[:200]}..." for k, v in analyses.items()])
    prompt = f"Business: {idea}\nRegion: {region}\n\nAnalyses:\n{analyses_summary}\n\nProvide critical review."
    return _respond(instr, prompt, max_tokens=600)

@traceable(name="synthesizer_agent")
def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str, region: str = "Pakistan") -> str:
    instr = (
        "You are a management consultant. Create a final report with:\n"
        "1. Executive Summary (3-4 bullets with Go/No-Go recommendation)\n"
        "2. Key Insights (top 5 findings)\n"
        "3. Next Steps (3-5 actions)\n"
        "Use markdown formatting."
    )
    analyses_summary = "\n".join([f"{k}: {v[:150]}..." for k, v in analyses.items()])
    prompt = (
        f"Business: {idea}\nRegion: {region}\n\n"
        f"Key findings:\n{analyses_summary}\n\n"
        f"Critique: {critique[:200]}...\n\n"
        "Create final report."
    )
    return _respond(instr, prompt, effort="high", max_tokens=1200)
