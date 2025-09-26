import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="📊", layout="wide")

# ---- Secrets → env BEFORE imports that use them ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
if "model_all" in st.secrets:
    os.environ["MODEL_ALL"] = st.secrets["model_all"]
if "verbosity" in st.secrets:
    os.environ["VERBOSITY"] = st.secrets["verbosity"]
if "USE_SERPER" in st.secrets:
    os.environ["USE_SERPER"] = st.secrets["USE_SERPER"]
if "SERPER_API_KEY" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

st.title("Business Idea Analyzer — GPT-5 (Responses API + Serper)")
st.caption("Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer")

if not os.getenv("OPENAI_API_KEY"):
    with st.expander("🔑 Configure OpenAI API key"):
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("API key set for this session.")

idea = st.text_area("Your business idea", height=140, placeholder="Describe your idea...")
region = st.text_input("Target region / market", value="Pakistan")

run = st.button("Analyze Idea", type="primary")

if run:
    if not idea.strip():
        st.error("Please enter a business idea.")
        st.stop()
    try:
        st.subheader("1) Planner");               plan = agents.planner_agent(idea); st.write(plan)
        st.subheader("2) Market Analysis");       market = agents.market_analysis_agent(idea, region=region); st.write(market)
        st.subheader("3) Competition");           competition = agents.competition_analysis_agent(idea, region=region); st.write(competition)
        st.subheader("4) Financial Feasibility"); financial = agents.financial_feasibility_agent(idea, region=region); st.write(financial)
        st.subheader("5) Go-to-Market (GTM)");    gtm = agents.gtm_strategy_agent(idea, region=region); st.write(gtm)
        st.subheader("6) Risks");                 risks = agents.risks_analysis_agent(idea, region=region); st.write(risks)
        st.subheader("7) Critique");              analyses = {"Market": market, "Competition": competition, "Financial": financial, "GTM": gtm, "Risks": risks}
                                                  
        critic = agents.critic_agent(idea, analyses, region=region); st.write(critic)
        st.subheader("8) Final Report");          final_md = agents.synthesizer_agent(idea, analyses, critic, region=region); st.markdown(final_md)

        pdf_bytes = generate_analysis_pdf(
            idea=idea, plan=plan, market=market, competition=competition,
            financial=financial, gtm=gtm, risks=risks, critic=critic, final_report=final_md
        )
        if pdf_bytes:
            st.download_button("Download Analysis (PDF)", data=pdf_bytes,
                               file_name="business_idea_analysis.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"❌ OpenAI error: {repr(e)}")
        st.stop()
