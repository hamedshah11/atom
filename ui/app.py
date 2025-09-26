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
if "ENABLE_WEB_SEARCH" in st.secrets:
    os.environ["ENABLE_WEB_SEARCH"] = st.secrets["ENABLE_WEB_SEARCH"]

# make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

st.title("Business Idea Analyzer — GPT-5 (Responses API)")
st.caption("Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer")

# Key for local dev (if not in secrets)
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
        # 1) Planner
        st.subheader("1) Planner")
        plan = agents.planner_agent(idea)
        st.write(plan)

        # 2) Market
        st.subheader("2) Market Analysis")
        market = agents.market_analysis_agent(idea, region=region)
        st.write(market)

        # 3) Competition
        st.subheader("3) Competition")
        competition = agents.competition_analysis_agent(idea, region=region)
        st.write(competition)

        # 4) Financial Feasibility
        st.subheader("4) Financial Feasibility")
        financial = agents.financial_feasibility_agent(idea, region=region)
        st.write(financial)

        # 5) GTM
        st.subheader("5) Go-to-Market (GTM)")
        gtm = agents.gtm_strategy_agent(idea, region=region)
        st.write(gtm)

        # 6) Risks
        st.subheader("6) Risks")
        risks = agents.risks_analysis_agent(idea, region=region)
        st.write(risks)

        # 7) Critique
        st.subheader("7) Critique")
        analyses = {
            "Market": market,
            "Competition": competition,
            "Financial": financial,
            "GTM": gtm,
            "Risks": risks,
        }
        critic = agents.critic_agent(idea, analyses, region=region)
        st.write(critic)

        # 8) Final Report
        st.subheader("8) Final Report")
        final_md = agents.synthesizer_agent(idea, analyses, critic, region=region)
        st.markdown(final_md)

        # PDF
        pdf_bytes = generate_analysis_pdf(
            idea=idea, plan=plan, market=market, competition=competition,
            financial=financial, gtm=gtm, risks=risks, critic=critic, final_report=final_md
        )
        if pdf_bytes:
            st.download_button("Download Analysis (PDF)", data=pdf_bytes,
                               file_name="business_idea_analysis.pdf", mime="application/pdf")
    except Exception as e:
        # show full error so we can debug on Streamlit Cloud
        st.error(f"❌ OpenAI error: {repr(e)}")
        st.stop()
