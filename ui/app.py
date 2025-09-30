import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="📊", layout="wide")
os.environ["SERPER_API_KEY"] = "e62e1e8919e57b754e3acd05b2d2bb570effb93e"
os.environ["USE_SERPER"] = "1"
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

st.title("Business Idea Analyzer — GPT-4 Turbo")
st.caption("Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer")

# Show current model
model_name = os.getenv("MODEL_ALL", "gpt-4-turbo")
st.info(f"Using model: {model_name}")

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
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please configure your OpenAI API key first.")
        st.stop()
    
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize results storage
        results = {}
        
        # 1) Planner
        status_text.text("🔄 Running Planner Agent...")
        with st.spinner("Planning analysis steps..."):
            plan = agents.planner_agent(idea)
            results['plan'] = plan
        st.subheader("1) Planner")
        st.write(plan)
        progress_bar.progress(12)
        
        # 2) Market Analysis
        status_text.text("🔄 Running Market Analysis Agent...")
        with st.spinner("Analyzing market size and opportunity..."):
            market = agents.market_analysis_agent(idea, region=region)
            results['market'] = market
        st.subheader("2) Market Analysis")
        st.write(market)
        progress_bar.progress(25)
        
        # 3) Competition
        status_text.text("🔄 Running Competition Analysis Agent...")
        with st.spinner("Analyzing competitive landscape..."):
            competition = agents.competition_analysis_agent(idea, region=region)
            results['competition'] = competition
        st.subheader("3) Competition")
        st.write(competition)
        progress_bar.progress(38)
        
        # 4) Financial Feasibility
        status_text.text("🔄 Running Financial Analysis Agent...")
        with st.spinner("Evaluating financial feasibility..."):
            financial = agents.financial_feasibility_agent(idea, region=region)
            results['financial'] = financial
        st.subheader("4) Financial Feasibility")
        st.write(financial)
        progress_bar.progress(50)
        
        # 5) GTM
        status_text.text("🔄 Running Go-to-Market Agent...")
        with st.spinner("Developing GTM strategy..."):
            gtm = agents.gtm_strategy_agent(idea, region=region)
            results['gtm'] = gtm
        st.subheader("5) Go-to-Market (GTM)")
        st.write(gtm)
        progress_bar.progress(62)
        
        # 6) Risks
        status_text.text("🔄 Running Risk Analysis Agent...")
        with st.spinner("Identifying risks and mitigations..."):
            risks = agents.risks_analysis_agent(idea, region=region)
            results['risks'] = risks
        st.subheader("6) Risks")
        st.write(risks)
        progress_bar.progress(75)
        
        # 7) Critique
        status_text.text("🔄 Running Critique Agent...")
        analyses = {
            "Market": market, 
            "Competition": competition, 
            "Financial": financial, 
            "GTM": gtm, 
            "Risks": risks
        }
        with st.spinner("Critically reviewing all analyses..."):
            critic = agents.critic_agent(idea, analyses, region=region)
            results['critic'] = critic
        st.subheader("7) Critique")
        st.write(critic)
        progress_bar.progress(88)
        
        # 8) Final Report
        status_text.text("🔄 Running Synthesis Agent...")
        with st.spinner("Synthesizing final report..."):
            final_md = agents.synthesizer_agent(idea, analyses, critic, region=region)
            results['final_report'] = final_md
        st.subheader("8) Final Report")
        st.markdown(final_md)
        progress_bar.progress(100)
        
        # Clear status and show completion
        status_text.text("✅ Analysis complete!")
        
        # Generate PDF
        with st.spinner("Generating PDF report..."):
            pdf_bytes = generate_analysis_pdf(
                idea=idea, 
                plan=plan, 
                market=market, 
                competition=competition,
                financial=financial, 
                gtm=gtm, 
                risks=risks, 
                critic=critic, 
                final_report=final_md
            )
        
        if pdf_bytes:
            st.download_button(
                "Download Analysis (PDF)", 
                data=pdf_bytes,
                file_name="business_idea_analysis.pdf", 
                mime="application/pdf"
            )
        
        # Show summary stats
        st.success("✅ Analysis completed successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Agents Run", "8")
        with col2:
            st.metric("Target Region", region)
        with col3:
            st.metric("Report Sections", "8")

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.error(f"Details: {repr(e)}")
        st.info("Common issues:\n- Invalid API key\n- Rate limits exceeded\n- Network issues\n- Model not available")
        st.stop()
