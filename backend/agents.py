import os
from typing import Dict, Optional
from openai import OpenAI, BadRequestError

# Models
GPT4_1 = "gpt-4.1"   # planner / critic / synthesizer
O4_MINI = "o4-mini"  # analysts (fast / cost-optimized)

# Lazy singleton client
_client: Optional[OpenAI] = None
def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def _chat(model: str, system_prompt: str, user_prompt: str, temperature: Optional[float] = None) -> str:
    """
    Wrapper for OpenAI Chat Completions (v1.x) that:
    - Adds temperature only if provided
    - If API complains about unsupported temperature, retries without it
    """
    client = get_client()

    def _make(kwargs_extra=None):
        base_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
        }
        if kwargs_extra:
            base_kwargs.update(kwargs_extra)
        resp = client.chat.completions.create(**base_kwargs)
        return (resp.choices[0].message.content or "").strip()

    # Try with temperature if given
    if temperature is not None:
        try:
            return _make({"temperature": temperature})
        except BadRequestError as e:
            # If model rejects temperature, fall back to default (omit it)
            if "Unsupported value" in str(e) and "temperature" in str(e):
                return _make()
            raise
    else:
        # No temperature requested
        return _make()

def planner_agent(idea: str) -> str:
    sys = ("You are a planning agent. Produce a concise ordered list of steps to analyze the business idea "
           "(market, competition, financials, GTM, risks). Keep it to 4–8 bullets.")
    usr = f"Business Idea:\n{idea}\n\nCreate the analysis plan."
    # gpt-4.1 usually supports temperature; ok to request a low one
    return _chat(GPT4_1, sys, usr, temperature=0.1)

def market_analysis_agent(idea: str) -> str:
    sys = ("You are a market sizing analyst. Estimate TAM, SAM, SOM with clear method & assumptions. "
           "End with a line like: TAM: X, SAM: Y, SOM: Z (include units).")
    usr = f"Business Idea:\n{idea}\n\nProvide market overview and TAM/SAM/SOM."
    # o4-mini may not support temperature → wrapper will retry without it
    return _chat(O4_MINI, sys, usr, temperature=0.2)

def competition_analysis_agent(idea: str) -> str:
    sys = ("You are a competition analyst. Identify 3–6 key competitors/alternatives, compare positioning, "
           "and summarize opportunities to differentiate.")
    usr = f"Business Idea:\n{idea}\n\nAnalyze the competitive landscape."
    return _chat(O4_MINI, sys, usr, temperature=0.2)

def financial_feasibility_agent(idea: str) -> str:
    sys = ("You are a finance analyst. Outline revenue model, pricing, COGS, gross margin, opex buckets, "
           "rough 3-year outlook, and breakeven logic. Separate assumptions vs. logic.")
    usr = f"Business Idea:\n{idea}\n\nEvaluate financial feasibility."
    return _chat(O4_MINI, sys, usr, temperature=0.2)

def gtm_strategy_agent(idea: str) -> str:
    sys = ("You are a GTM strategist. Define ICPs, channels, key messages, a 90-day launch plan, and core KPIs. "
           "Add a simple funnel (impressions→leads→conversions) with baseline assumptions.")
    usr = f"Business Idea:\n{idea}\n\nPropose GTM strategy."
    return _chat(O4_MINI, sys, usr, temperature=0.2)

def risks_analysis_agent(idea: str) -> str:
    sys = ("You are a risk analyst. List major risks (regulatory, technical, market, execution, finance) with "
           "likelihood/impact and brief mitigations.")
    usr = f"Business Idea:\n{idea}\n\nIdentify key risks & mitigations."
    return _chat(O4_MINI, sys, usr, temperature=0.2)

def critic_agent(idea: str, analyses: Dict[str, str]) -> str:
    sys = ("You are a tough critic. Review the analyses for gaps, contradictions, over-optimism, or missing data. "
           "Return a bullet list of fixes and questions to validate.")
    blob = "\n\n".join([f"[{k}]\n{v}" for k, v in analyses.items()])
    usr = f"Business Idea:\n{idea}\n\nAnalyses:\n{blob}\n\nProvide critique."
    return _chat(GPT4_1, sys, usr, temperature=0.1)

def synthesizer_agent(idea: str, analyses: Dict[str, str], critique: str) -> str:
    sys = ("You are a precise management consultant. Synthesize into:\n"
           "1) LEAN CANVAS (Problem, Customer Segments, UVP, Solution, Channels, Revenue, Costs, Key Metrics, Unfair Advantage)\n"
           "2) MARKET ANALYSIS (TAM/SAM/SOM + method; competitor summary)\n"
           "3) FEASIBILITY (unit econ, breakeven, risks, go/no-go)\n"
           "Integrate the critique as caveats. Output Markdown.")
    blob = "\n\n".join([f"{k} Analysis:\n{v}" for k, v in analyses.items()])
    usr = (f"Business Idea:\n{idea}\n\nInputs:\n{blob}\n\nCritique:\n{critique}\n\n"
           "Produce the final consolidated report in Markdown.")
    return _chat(GPT4_1, sys, usr, temperature=0.1)
