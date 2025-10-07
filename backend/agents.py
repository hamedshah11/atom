import os
import json
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from langsmith import traceable
from backend.retry_wrapper import retry_on_failure

# ===== Model Configuration =====
AVAILABLE_MODELS = ["o3-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
DEFAULT_MODEL = "gpt-4o"

VERBOSITY_ALL = os.getenv("VERBOSITY", "low")
USE_SERPER = os.getenv("USE_SERPER", "0") == "1"
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

# Rate limiting - INCREASED for reliability
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "2.5"))

# LangSmith configuration
LANGCHAIN_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment or Streamlit secrets")
        _client = OpenAI(api_key=api_key, timeout=60.0, max_retries=3)
    return _client

def _debug_log(message: str):
    """Print debug messages if debug mode is on"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def _get_current_model() -> str:
    """Get the current model from environment - DYNAMICALLY"""
    model = os.getenv("MODEL_ALL", DEFAULT_MODEL)
    if model not in AVAILABLE_MODELS:
        _debug_log(f"Warning: Model {model} not available. Using {DEFAULT_MODEL}")
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
        _debug_log(f"Making API call with model: {kwargs.get('model', 'unknown')}")
        response = get_client().chat.completions.create(**kwargs)
        _debug_log(f"API call successful")
        return response
    except Exception as e:
        error_msg = str(e)
        _debug_log(f"API Error: {type(e).__name__}: {error_msg}")
        
        # Provide more specific error messages
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            raise RuntimeError("Authentication failed. Check your OPENAI_API_KEY.")
        elif "rate_limit" in error_msg.lower():
            raise RuntimeError("Rate limit exceeded. Increase RATE_LIMIT_DELAY in .env")
        elif "timeout" in error_msg.lower():
            raise RuntimeError("Request timed out. Check your network connection.")
        else:
            raise

def _safe_output_text(resp: Any) -> str:
    """Extract text from OpenAI chat completion response"""
    if resp and hasattr(resp, 'choices') and resp.choices:
        if hasattr(resp.choices[0], 'message') and hasattr(resp.choices[0].message, 'content'):
            content = resp.choices[0].message.content
            if content:
                return content.strip()
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
    OpenAI Chat API helper optimized for reliability
    Falls back to gpt-4o if primary model fails
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
    
    _debug_log(f"Using model: {model}, max_tokens: {max_tokens}")
    
    # Try with primary model first
    try:
        return _call_model(instructions, prompt, model, effort, verbosity, max_tokens)
    except Exception as e:
        error_msg = str(e)
        _debug_log(f"Primary model failed: {error_msg}")
        
        # If o3-mini fails, fall back to gpt-4o
        if model == "o3-mini" and "gpt-4o" in AVAILABLE_MODELS:
            if any(keyword in error_msg.lower() for keyword in ["tier", "quota", "does not exist", "not found"]):
                _debug_log(f"O3-mini unavailable, falling back to gpt-4o")
                try:
                    return _call_model(instructions, prompt, "gpt-4o", effort, verbosity, max_tokens)
                except Exception as fallback_error:
                    _debug_log(f"Fallback also failed: {str(fallback_error)}")
                    return f"⚠️ Both o3-mini and gpt-4o failed. Error: {str(fallback_error)[:200]}"
        
        # For other errors, return formatted error message
        if "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
            return "⚠️ API Authentication Error: Please check your OPENAI_API_KEY"
        elif "rate limit" in error_msg.lower():
            return "⚠️ Rate limit exceeded. Please increase RATE_LIMIT_DELAY and try again."
        elif "timeout" in error_msg.lower():
            return "⚠️ Request timeout. Check network connection or try again."
        else:
            return f"⚠️ API Error: {error_msg[:250]}"

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
    
    # O3-mini configuration
    if is_o3_mini:
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
            "max_completion_tokens": max_tokens,
        }
        
        # O3-mini supports reasoning_effort
        reasoning_map = {"minimal": "low", "low": "low", "medium": "medium", "high": "high"}
        api_params["reasoning_effort"] = reasoning_map.get(effort, "medium")
    
    else:
        # GPT-4o and other models configuration
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
        _debug_log(f"Response length: {len(text)} chars")
        return text
    else:
        raise Exception("No response content generated from API")

# ====================== Serper helpers ======================
def _format_serper_block(results: list[dict], label: str = "Market Research") -> str:
    """Format Serper search results for context"""
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
@retry_on_failure(max_retries=2, delay=3.0)
def planner_agent(idea: str) -> str:
    """Create strategic analysis plan"""
    instr = (
        "You are a strategic planning agent. Create 4-6 concise steps to analyze this business idea. "
        "Focus on: market size, competition, financials, go-to-market, and risks. "
        "Be specific and actionable."
    )
    prompt = f"Business Idea: {idea}\n\nCreate a brief strategic analysis plan."
    return _respond(instr, prompt, effort="minimal", max_tokens=500)

@traceable(name="market_analysis_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def market_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    """Analyze market size and potential"""
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} {idea.split()[0]} market size statistics trends"
            results = web_search_serper(q, num=3)
            serper_block = _format_serper_block(results, "Market Research Data")
        except Exception as e:
            _debug_log(f"Serper search failed: {e}")
            pass

    instr = (
        "You are a market analyst. Estimate TAM, SAM, and SOM for this business. "
        f"Focus on the {region} market specifically. "
        "Format your response as:\n"
        "1) Brief market assumptions (2-3 sentences)\n"
        "2) Simple market sizing table\n"
        "3) Final summary line: TAM: X, SAM: Y, SOM: Z\n\n"
        "Be realistic and data-driven."
    )
    prompt = f"Business: {idea}\nRegion: {region}{serper_block}\n\nProvide market analysis."
    return _respond(instr, prompt, effort="medium", max_tokens=900)

@traceable(name="competition_analysis_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def competition_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    """Analyze competitive landscape"""
    serper_block = ""
    if USE_SERPER:
        try:
            from backend.serper import web_search_serper
            q = f"{region} {idea.split()[0]} competitors companies market leaders"
            serper_block = _format_serper_block(web_search_serper(q, num=3), "Competitor Research")
        except Exception as e:
            _debug_log(f"Serper search failed: {e}")
            pass
    
    instr = (
        "You are a competition analyst. Identify 3-5 key competitors in the specified region. "
        "For each competitor, provide:\n"
        "- Name and brief description\n"
        "- Market positioning\n"
        "- Key strengths\n\n"
        "Then suggest differentiation opportunities."
    )
    prompt = f"Business: {idea}\nRegion: {region}{serper_block}\n\nAnalyze competition."
    return _respond(instr, prompt, effort="medium", max_tokens=800)

@traceable(name="financial_feasibility_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def financial_feasibility_agent(idea: str, region: str = "Pakistan") -> str:
    """Analyze financial feasibility"""
    instr = (
        "You are a financial analyst. Create a financial feasibility analysis with:\n"
        "1) Revenue model and pricing strategy\n"
        "2) Major cost categories (startup and ongoing)\n"
        "3) Estimated gross margins\n"
        "4) Simple 3-year financial projection\n"
        f"5) Use local currency for {region}\n\n"
        "Be realistic and conservative in estimates."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nProvide financial analysis."
    return _respond(instr, prompt, effort="medium", max_tokens=900)

@traceable(name="gtm_strategy_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def gtm_strategy_agent(idea: str, region: str = "Pakistan") -> str:
    """Create go-to-market strategy"""
    instr = (
        "You are a GTM strategist. Create a go-to-market strategy with:\n"
        "1) Target customer segments (be specific)\n"
        "2) Marketing channels (relevant to the region)\n"
        "3) Key messaging and value proposition\n"
        "4) 90-day launch plan with 5-7 concrete action items\n\n"
        "Be practical and region-specific."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nCreate GTM strategy."
    return _respond(instr, prompt, effort="medium", max_tokens=800)

@traceable(name="risks_analysis_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def risks_analysis_agent(idea: str, region: str = "Pakistan") -> str:
    """Identify and analyze risks"""
    instr = (
        "You are a risk analyst. Identify 5 major risks with:\n"
        "- Risk category (regulatory, market, operational, financial, competitive)\n"
        "- Impact level (High/Medium/Low)\n"
        "- Likelihood (High/Medium/Low)\n"
        "- Mitigation strategy\n\n"
        f"Focus on risks specific to {region}."
    )
    prompt = f"Business: {idea}\nRegion: {region}\n\nIdentify key risks."
    return _respond(instr, prompt, effort="medium", max_tokens=700)

@traceable(name="critic_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def critic_agent(idea: str, analyses: Dict[str, str], region: str = "Pakistan") -> str:
    """Critical review of analyses"""
    instr = (
        "You are a critical business consultant. Review the analyses and identify:\n"
        "1) Major gaps or missing information\n"
        "2) Contradictions or inconsistencies\n"
        "3) Unrealistic assumptions\n"
        "4) Areas needing more research\n"
        "5) Critical concerns that could derail the business\n\n"
        "Be honest and thorough."
    )
    analyses_summary = "\n\n".join([f"## {k}\n{v[:300]}..." for k, v in analyses.items() if v])
    prompt = f"Business: {idea}\nRegion: {region}\n\nAnalyses:\n{analyses_summary}\n\nProvide critical review."
    return _respond(instr, prompt, effort="medium", max_tokens=700)

@traceable(name="synthesizer_agent")
@retry_on_failure(max_retries=2, delay=3.0)
def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str, region: str = "Pakistan") -> str:
    """Synthesize final executive report"""
    instr = (
        "You are a management consultant. Create a final executive report with:\n\n"
        "## Executive Summary\n"
        "- 3-4 bullet points summarizing key findings\n"
        "- Clear Go/No-Go/Conditional-Go recommendation with reasoning\n\n"
        "## Key Insights\n"
        "- Top 5 most important findings across all analyses\n\n"
        "## Critical Success Factors\n"
        "- 3-4 factors that will determine success\n\n"
        "## Next Steps\n"
        "- 3-5 concrete actions to take\n\n"
        "Use markdown formatting. Be decisive and actionable."
    )
    analyses_summary = "\n\n".join([f"## {k}\n{v[:250]}..." for k, v in analyses.items() if v])
    prompt = (
        f"Business: {idea}\nRegion: {region}\n\n"
        f"Key Analyses:\n{analyses_summary}\n\n"
        f"Critical Review:\n{critique[:300]}...\n\n"
        "Create comprehensive final report."
    )
    return _respond(instr, prompt, effort="high", max_tokens=1400)
