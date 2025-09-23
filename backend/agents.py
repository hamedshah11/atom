import os
import openai

# Retrieve OpenAI API key from environment variable (or configure in Streamlit secrets).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Model names for GPT-4.1 and o4-mini
GPT4_MODEL = "gpt-4"        # Placeholder for GPT-4.1 model
O4MINI_MODEL = "gpt-3.5-turbo"  # Placeholder for o4-mini model (using GPT-3.5 as example)

def call_openai_chat(model: str, system_prompt: str, user_prompt: str) -> str:
    """
    Helper to call OpenAI ChatCompletion (using updated OpenAI API) and return the assistant message text.
    """
    try:
        # Use the new OpenAI chat completion call (v1.0.0+)
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ]
            # You can include other params like temperature, max_tokens, etc., as needed
        )
        # Extract the assistant's reply using attribute access (since response is an object in v1+)
        answer_text = response.choices[0].message.content
        return answer_text.strip()
    except Exception as e:
        # If there's an API error or missing key, return an error message
        return f"Error: {e}"

def planner_agent(idea: str) -> str:
    """
    Uses GPT-4.1 to generate a plan (list of analytical steps) for the given business idea.
    """
    system_prompt = ("You are a planning agent that helps break down a business idea into key analysis steps. "
                     "You will be given a business idea, and you need to output a clear plan outlining what "
                     "aspects should be analyzed to evaluate this idea. Keep the plan concise and in an ordered list.")
    user_prompt = f"Business Idea: {idea}\n\nGenerate an analysis plan covering all important aspects."
    return call_openai_chat(GPT4_MODEL, system_prompt, user_prompt)

def market_analysis_agent(idea: str) -> str:
    """
    Uses o4-mini to analyze market size (TAM, SAM, SOM) for the idea. Asks for numeric estimates and rationale.
    """
    system_prompt = ("You are a market analyst agent focusing on market sizing. Analyze the Total Addressable Market (TAM), "
                     "Serviceable Addressable Market (SAM), and Serviceable Obtainable Market (SOM) for the given business idea. "
                     "Provide reasoning and estimates for each. End your response with a summary line listing TAM, SAM, SOM values.")
    user_prompt = f"Business Idea: {idea}\n\nConduct a market size analysis (TAM, SAM, SOM) with estimates."
    return call_openai_chat(O4MINI_MODEL, system_prompt, user_prompt)

def competition_analysis_agent(idea: str) -> str:
    """
    Uses o4-mini to analyze the competition for the idea, listing key competitors and competitive landscape.
    """
    system_prompt = ("You are a competition analyst agent. Identify and analyze the current competition for the given business idea. "
                     "List key competitors or alternative solutions and briefly discuss how the idea can differentiate itself.")
    user_prompt = f"Business Idea: {idea}\n\nAnalyze the competitive landscape and main competitors."
    return call_openai_chat(O4MINI_MODEL, system_prompt, user_prompt)

def financial_feasibility_agent(idea: str) -> str:
    """
    Uses o4-mini to evaluate the financial feasibility of the idea, outlining costs, revenue model, and profitability potential.
    """
    system_prompt = ("You are a financial analyst agent. Assess the financial feasibility of the given business idea. "
                     "Consider startup costs, operating costs, revenue streams, pricing, and potential profitability. "
                     "Provide a brief financial outlook (e.g., break-even point, funding needed).")
    user_prompt = f"Business Idea: {idea}\n\nEvaluate the financial feasibility (costs, revenue model, profitability)."
    return call_openai_chat(O4MINI_MODEL, system_prompt, user_prompt)

def gtm_strategy_agent(idea: str) -> str:
    """
    Uses o4-mini to propose a go-to-market strategy for the idea, including target customers, marketing channels, and acquisition strategy.
    """
    system_prompt = ("You are a marketing strategist agent. For the given business idea, outline a go-to-market strategy. "
                     "Include target customer segments, key marketing and distribution channels, and tactics for initial customer acquisition.")
    user_prompt = f"Business Idea: {idea}\n\nOutline a go-to-market strategy."
    return call_openai_chat(O4MINI_MODEL, system_prompt, user_prompt)

def risks_analysis_agent(idea: str) -> str:
    """
    Uses o4-mini to identify key risks and challenges for the idea and possible mitigations.
    """
    system_prompt = ("You are a risk analyst agent. Identify the main risks, challenges, or uncertainties associated with the given business idea. "
                     "Consider market risks, execution risks, competition, regulatory, or financial risks, and suggest possible mitigations.")
    user_prompt = f"Business Idea: {idea}\n\nIdentify key risks and challenges, with potential mitigations."
    return call_openai_chat(O4MINI_MODEL, system_prompt, user_prompt)

def critic_agent(idea: str, analyses: dict) -> str:
    """
    Uses GPT-4.1 to critique the combined analyses. Points out any missing elements or weaknesses in the analyses.
    """
    system_prompt = ("You are a critical evaluator agent. You will receive a business idea and the analyses of various aspects of that idea (market, competition, financial, go-to-market, risks). "
                     "Review the analyses and point out any gaps, inconsistencies, or additional factors that should be considered. "
                     "Be constructive and list any missing pieces or questions that need to be addressed.")
    # Combine all analysis texts for the assistant to review.
    analyses_text = ""
    for key, text in analyses.items():
        analyses_text += f"\nAnalysis - {key}:\n{text}\n"
    user_prompt = f"Business Idea: {idea}\n{analyses_text}\nProvide a critique of the above analyses."
    return call_openai_chat(GPT4_MODEL, system_prompt, user_prompt)

def synthesizer_agent(idea: str, analyses: dict, critique: str) -> str:
    """
    Uses GPT-4.1 to synthesize a final report from all analyses and critique.
    Outputs a comprehensive business report including a Lean Canvas, Market Analysis, and Feasibility.
    """
    system_prompt = ("You are a synthesis agent. You will compile a comprehensive business analysis report based on a business idea, several analytical outputs (market, competition, financial, go-to-market, risks), and a critique. "
                     "Your report should include:\n"
                     "- A Lean Canvas summary of the idea (cover key points like problem, solution, unique value, customer segments, revenue streams, cost structure, etc.).\n"
                     "- A Market Analysis section (summarize TAM/SAM/SOM and market characteristics).\n"
                     "- A Financial Feasibility section (startup costs, revenue potential, profitability timeline).\n"
                     "Integrate insights from the competition, go-to-market, and risks analyses as well. "
                     "End with a conclusion on viability. Use clear headings and concise language.")
    analyses_text = ""
    for key, text in analyses.items():
        analyses_text += f"\n{key} Analysis:\n{text}\n"
    # Include critique as well if available
    if critique:
        analyses_text += f"\nCritic Feedback:\n{critique}\n"
    user_prompt = f"Business Idea: {idea}\n\nGenerate a final report based on all the above information."
    return call_openai_chat(GPT4_MODEL, system_prompt, user_prompt)
