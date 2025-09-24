import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="📊", layout="wide")

# --- Secrets → env before imports ---
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
if "model_all" in st.secrets:
    os.environ["MODEL_ALL"] = st.secrets["model_all"]

# Repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from backend.run_graph import run_full
from utils.pdf import generate_analysis_pdf

st.title("Business Idea Analyzer — GPT-5 (Responses API)")
st.caption("Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer")

# Local dev key input (if secrets not set)
if not os.getenv("OPENAI_API_KEY"):
    with st.expander("🔑 Configure OpenAI API key"):
        key = st.text_input("OpenAI API Key", type="password")
        if key:
            os.environ["OPENAI_API_KEY"] = key
            st.success("API key set for this session.")

idea = st.text_area("Your business idea", height=140, placeholder="Describe your idea...")

colA, colB = st.columns(2)
with colA:
    run_seq = st.button("Analyze (sequential agents)", type="primary")
with colB:
    run_graph_btn = st.button("Analyze (LangGraph pipeline)")

def _maybe_warn_fallback(models_dict):
    if not models_dict:
        return
    used = set(models_dict.values())
    if "gpt-5" not in used:
        st.warning("Model fallback in effect (no GPT-5 access). "
                   "You can set `model_all = \"gpt-5\"` in secrets if your key supports it.")
    elif len(used) > 1:
        st.info("Mixed models used this run: " + ", ".join(sorted(used)))

if (run_seq or run_graph_btn):
    if not idea.strip():
        st.error("Please enter a business idea.")
        st.stop()

    try:
        if run_seq:
            st.subheader("1) Planner")
            plan, m1 = agents.planner_agent(idea); st.write(plan)

            st.subheader("2) Market Analysis")
            market, m2 = agents.market_analysis_agent(idea); st.write(market)

            st.subheader("3) Competition")
            competition, m3 = agents.competition_analysis_agent(idea); st.write(competition)

            st.subheader("4) Financial Feasibility")
            financial, m4 = agents.financial_feasibility_agent(idea); st.write(financial)

            st.subheader("5) Go-to-Market (GTM)")
            gtm, m5 = agents.gtm_strategy_agent(idea); st.write(gtm)

            st.subheader("6) Risks")
            risks, m6 = agents.risks_analysis_agent(idea); st.write(risks)

            st.subheader("7) Critique")
            analyses = {
                "Market": market, "Competition": competition, "Financial": financial, "GTM": gtm, "Risks": risks
            }
            critic, m7 = agents.critic_agent(idea, analyses); st.write(critic)

            st.subheader("8) Final Report")
            final_md, m8 = agents.synthesizer_agent(idea, analyses, critic); st.markdown(final_md)

            _maybe_warn_fallback({
                "planner": m1, "market": m2, "competition": m3, "financial": m4,
                "gtm": m5, "risks": m6, "critic": m7, "synth": m8
            })

            pdf_bytes = generate_analysis_pdf(
                idea=idea, plan=plan, market=market, competition=competition,
                financial=financial, gtm=gtm, risks=risks, critic=critic, final_report=final_md
            )
            if pdf_bytes:
                st.download_button("Download Analysis (PDF)", data=pdf_bytes,
                                   file_name="business_idea_analysis.pdf", mime="application/pdf")

        else:
            final_state = run_full(idea)
            plan        = final_state.get("plan", "")
            market      = final_state.get("market", "")
            competition = final_state.get("competition", "")
            financial   = final_state.get("financial", "")
            gtm         = final_state.get("gtm", "")
            risks       = final_state.get("risks", "")
            critic      = final_state.get("critic", "")
            final_md    = final_state.get("final_report", "")
            models_used = final_state.get("models", {})

            st.subheader("1) Planner");               st.write(plan)
            st.subheader("2) Market Analysis");       st.write(market)
            st.subheader("3) Competition");           st.write(competition)
            st.subheader("4) Financial Feasibility"); st.write(financial)
            st.subheader("5) Go-to-Market (GTM)");    st.write(gtm)
            st.subheader("6) Risks");                 st.write(risks)
            st.subheader("7) Critique");              st.write(critic)
            st.subheader("8) Final Report");          st.markdown(final_md)

            _maybe_warn_fallback(models_used)

            pdf_bytes = generate_analysis_pdf(
                idea=idea, plan=plan, market=market, competition=competition,
                financial=financial, gtm=gtm, risks=risks, critic=critic, final_report=final_md
            )
            if pdf_bytes:
                st.download_button("Download Analysis (PDF)", data=pdf_bytes,
                                   file_name="business_idea_analysis.pdf", mime="application/pdf")

    except Exception as e:
        st.error(f"❌ OpenAI error: {e}")
        st.stop()
