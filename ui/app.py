import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="📊", layout="wide")

# --- Streamlit secrets → environment BEFORE importing backend ---
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
# Optional: allow overriding model via secrets (lowercase keys in secrets.toml)
if "model_all" in st.secrets:
    os.environ["MODEL_ALL"] = st.secrets["model_all"]

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.visuals import make_market_chart, draw_agent_graph
from utils.pdf import generate_analysis_pdf

st.title("Business Idea Analyzer")
st.write(
    "Enter a business idea. This app runs multiple specialist AI agents "
    "(Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer) "
    "and returns a Lean Canvas, Market Analysis (with TAM/SAM/SOM), and Feasibility."
)

# Fallback: local dev key entry
if not os.getenv("OPENAI_API_KEY"):
    with st.expander("🔑 Configure OpenAI API key"):
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("API key set for this session.")

idea = st.text_area("Your business idea", height=140, placeholder="Describe your idea...")

run = st.button("Analyze Idea", type="primary")

if run:
    if not idea.strip():
        st.error("Please enter a business idea.")
        st.stop()

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("1) Planner")
        plan = agents.planner_agent(idea)
        st.write(plan)

        st.subheader("2) Market Analysis")
        market = agents.market_analysis_agent(idea)
        st.write(market)

        st.subheader("3) Competition")
        competition = agents.competition_analysis_agent(idea)
        st.write(competition)

        st.subheader("4) Financial Feasibility")
        financial = agents.financial_feasibility_agent(idea)
        st.write(financial)

        st.subheader("5) Go-to-Market (GTM)")
        gtm = agents.gtm_strategy_agent(idea)
        st.write(gtm)

        st.subheader("6) Risks")
        risks = agents.risks_analysis_agent(idea)
        st.write(risks)

        st.subheader("7) Critique")
        analyses = {
            "Market": market,
            "Competition": competition,
            "Financial": financial,
            "GTM": gtm,
            "Risks": risks,
        }
        critic = agents.critic_agent(idea, analyses)
        st.write(critic)

        st.subheader("8) Final Report")
        final_md = agents.synthesizer_agent(idea, analyses, critic)
        st.markdown(final_md)

    with col2:
        st.subheader("Visuals")
        chart_path = make_market_chart(market, filename="tam_sam_som_chart.png")
        if chart_path:
            st.image(chart_path, caption="TAM / SAM / SOM")

        graph_path = draw_agent_graph(filename="agent_workflow.png")
        if graph_path:
            st.image(graph_path, caption="Agent Workflow")

        st.divider()
        pdf_bytes = generate_analysis_pdf(
            idea=idea, plan=plan, market=market, competition=competition,
            financial=financial, gtm=gtm, risks=risks, critic=critic,
            final_report=final_md, chart_path=chart_path, graph_path=graph_path
        )
        if pdf_bytes:
            st.download_button("Download Full Analysis (PDF)", data=pdf_bytes,
                               file_name="business_idea_analysis.pdf", mime="application/pdf")
        else:
            st.caption("PDF generation unavailable.")
