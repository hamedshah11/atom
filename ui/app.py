import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="📊", layout="wide")

# Configure Serper
os.environ["SERPER_API_KEY"] = "e62e1e8919e57b754e3acd05b2d2bb570effb93e"
os.environ["USE_SERPER"] = "1"

# ---- Secrets → env BEFORE imports ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
if "model_all" in st.secrets:
    os.environ["MODEL_ALL"] = st.secrets["model_all"]
if "verbosity" in st.secrets:
    os.environ["VERBOSITY"] = st.secrets["verbosity"]

# LangSmith configuration
if "langchain_api_key" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_key"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("langchain_tracing_v2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
if "langchain_project" in st.secrets:
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["langchain_project"]
else:
    os.environ["LANGCHAIN_PROJECT"] = "business-idea-analyzer"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

st.title("Business Idea Analyzer — AI-Powered Analysis")
st.caption("Planner → Market → Competition → Financials → GTM → Risks → Critic → Synthesizer")

# Check configurations
langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "business-idea-analyzer")
model_all = os.getenv("MODEL_ALL", "gpt-4o-mini")

# Status bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    model_emoji = {
        "gpt-4o": "🌟",
        "gpt-4o-mini": "⚡",
        "o1": "🧠",
        "o3-mini": "🤔",
        "o1-preview": "⚠️",  # Legacy
        "o1-mini": "⚠️",     # Legacy
        "gpt-4-turbo": "❌",
        "gpt-3.5-turbo": "❌"
    }
    emoji = model_emoji.get(model_all, "🤖")
    st.info(f"{emoji} Model: {model_all}")
    
    if model_all in ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
        st.error("❌ This model is deprecated! Use gpt-4o or gpt-4o-mini")
    elif model_all in ["o1-preview", "o1-mini"]:
        st.warning("⚠️ Legacy model! Use o1 or o3-mini instead")

with col2:
    if langsmith_enabled:
        st.success("🔍 LangSmith: ON")
    else:
        st.info("🔍 LangSmith: OFF")

with col3:
    st.info(f"🔎 Search: ON")

# Debug sidebar
with st.sidebar:
    with st.expander("🐛 Debug Info", expanded=False):
        st.write("**LangSmith Status:**")
        st.write(f"API Key: {'✅ Set' if os.getenv('LANGCHAIN_API_KEY') else '❌ Missing'}")
        if os.getenv('LANGCHAIN_API_KEY'):
            key = os.getenv('LANGCHAIN_API_KEY')
            st.write(f"Prefix: {key[:10]}...")
        st.write(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
        st.write(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
        st.write(f"Endpoint: {os.getenv('LANGCHAIN_ENDPOINT')}")
        
        st.divider()
        
        if st.button("🧪 Test LangSmith"):
            try:
                from langsmith import Client
                api_key = os.getenv("LANGCHAIN_API_KEY")
                if not api_key:
                    st.error("❌ No API key!")
                else:
                    with st.spinner("Testing..."):
                        client = Client(api_key=api_key)
                        projects = list(client.list_projects(limit=3))
                        st.success(f"✅ Connected! {len(projects)} projects")
                        for p in projects:
                            st.write(f"- {p.name}")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)[:100]}")

# Configuration panel
with st.expander("⚙️ Configuration", expanded=False):
    tab1, tab2, tab3 = st.tabs(["OpenAI", "Models", "LangSmith"])
    
    with tab1:
        if not os.getenv("OPENAI_API_KEY"):
            key = st.text_input("OpenAI API Key", type="password")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                st.success("✅ Set for session")
        else:
            st.success("✅ OpenAI configured")
    
    with tab2:
        st.subheader("🤖 Model Selection")
        
        # Model info with updated details
        st.write("**Available Models (October 2025):**")
        
        models_info = {
            "gpt-4o-mini": {
                "emoji": "⚡", 
                "cost": "$0.02", 
                "speed": "★★★★★", 
                "quality": "★★★★☆", 
                "desc": "Best value",
                "note": "Recommended for daily use"
            },
            "gpt-4o": {
                "emoji": "🌟", 
                "cost": "$0.08", 
                "speed": "★★★★☆", 
                "quality": "★★★★★", 
                "desc": "Best quality",
                "note": "Premium performance"
            },
            "o1": {
                "emoji": "🧠", 
                "cost": "$0.50", 
                "speed": "★★☆☆☆", 
                "quality": "★★★★★", 
                "desc": "Advanced reasoning",
                "note": "Requires API Tier 1+ ($5 spend)"
            },
            "o3-mini": {
                "emoji": "🤔", 
                "cost": "$0.15", 
                "speed": "★★★☆☆", 
                "quality": "★★★★☆", 
                "desc": "Efficient reasoning",
                "note": "Requires API Tier 1+ ($5 spend)"
            }
        }
        
        # Current model status
        current_model = os.getenv("MODEL_ALL", "gpt-4o-mini")
        if current_model in models_info:
            current_info = models_info[current_model]
            st.success(f"**Active:** {current_info['emoji']} {current_model} — {current_info['cost']}/analysis")
        else:
            st.warning(f"⚠️ Active: {current_model} (legacy/unknown)")
        
        st.divider()
        
        # Model selection cards
        cols = st.columns(2)
        
        for idx, (model, info) in enumerate(models_info.items()):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.write(f"### {info['emoji']} {model}")
                    st.write(f"**{info['desc']}**")
                    st.write(f"💰 **{info['cost']}**/analysis")
                    st.write(f"⚡ Speed: {info['speed']}")
                    st.write(f"✨ Quality: {info['quality']}")
                    st.caption(f"ℹ️ {info['note']}")
                    
                    if st.button(f"Use {model}", key=f"select_{model}", 
                               disabled=(model == current_model),
                               use_container_width=True):
                        os.environ["MODEL_ALL"] = model
                        st.rerun()
        
        st.divider()
        
        # Legacy models warning
        with st.expander("⚠️ Legacy Models (Not Recommended)", expanded=False):
            st.warning("**These models have been replaced:**\n"
                      "- ❌ o1-preview → Use **o1** instead\n"
                      "- ❌ o1-mini → Use **o3-mini** instead\n"
                      "- ❌ gpt-4-turbo → Use **gpt-4o** instead\n"
                      "- ❌ gpt-3.5-turbo → Use **gpt-4o-mini** instead")
    
    with tab3:
        st.subheader("LangSmith Observability")
        
        if not langsmith_enabled:
            st.info("Enable to track executions\n\nGet key: https://smith.langchain.com/settings")
            
            key = st.text_input("LangSmith API Key", type="password", 
                               help="Get from https://smith.langchain.com/settings")
            proj = st.text_input("Project", value="business-idea-analyzer")
            
            if st.button("Enable LangSmith", type="primary"):
                if key:
                    if not key.startswith(('lsv2_', 'ls__')):
                        st.warning("⚠️ Key format looks unusual")
                    
                    os.environ["LANGCHAIN_API_KEY"] = key
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = proj
                    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                    st.success("✅ Enabled! Restarting...")
                    st.rerun()
                else:
                    st.error("Enter API key")
        else:
            st.success("✅ LangSmith enabled")
            st.write(f"**Project:** {langsmith_project}")
            st.write("[View Dashboard](https://smith.langchain.com/)")
            
            if st.button("Disable"):
                os.environ["LANGCHAIN_TRACING_V2"] = "false"
                st.rerun()

# Main input
idea = st.text_area("Your business idea", height=140, placeholder="Describe your idea...")
region = st.text_input("Target region", value="Pakistan")

run = st.button("Analyze Idea", type="primary")

if run:
    if not idea.strip():
        st.error("Enter a business idea")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Configure OpenAI API key first")
        st.stop()
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Planner
        status_text.text("🔄 Planner Agent...")
        with st.spinner("Planning..."):
            plan = agents.planner_agent(idea)
        st.subheader("1) Planner")
        st.write(plan)
        progress_bar.progress(12)
        
        # Market
        status_text.text("🔄 Market Analysis...")
        with st.spinner("Analyzing market..."):
            market = agents.market_analysis_agent(idea, region)
        st.subheader("2) Market Analysis")
        st.write(market)
        progress_bar.progress(25)
        
        # Competition
        status_text.text("🔄 Competition Analysis...")
        with st.spinner("Analyzing competition..."):
            competition = agents.competition_analysis_agent(idea, region)
        st.subheader("3) Competition")
        st.write(competition)
        progress_bar.progress(38)
        
        # Financial
        status_text.text("🔄 Financial Analysis...")
        with st.spinner("Financial modeling..."):
            financial = agents.financial_feasibility_agent(idea, region)
        st.subheader("4) Financial Feasibility")
        st.write(financial)
        progress_bar.progress(50)
        
        # GTM
        status_text.text("🔄 Go-to-Market...")
        with st.spinner("GTM strategy..."):
            gtm = agents.gtm_strategy_agent(idea, region)
        st.subheader("5) Go-to-Market")
        st.write(gtm)
        progress_bar.progress(62)
        
        # Risks
        status_text.text("🔄 Risk Analysis...")
        with st.spinner("Identifying risks..."):
            risks = agents.risks_analysis_agent(idea, region)
        st.subheader("6) Risks")
        st.write(risks)
        progress_bar.progress(75)
        
        # Critique
        status_text.text("🔄 Critique...")
        analyses = {
            "Market": market,
            "Competition": competition,
            "Financial": financial,
            "GTM": gtm,
            "Risks": risks
        }
        with st.spinner("Critical review..."):
            critic = agents.critic_agent(idea, analyses, region)
        st.subheader("7) Critique")
        st.write(critic)
        progress_bar.progress(88)
        
        # Final
        status_text.text("🔄 Synthesis...")
        with st.spinner("Final report..."):
            final_md = agents.synthesizer_agent(idea, analyses, critic, region)
        st.subheader("8) Final Report")
        st.markdown(final_md)
        progress_bar.progress(100)
        
        status_text.text("✅ Complete!")
        
        # PDF
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_analysis_pdf(
                    idea, plan, market, competition, financial,
                    gtm, risks, critic, final_md
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 Download PDF",
                        data=pdf_bytes,
                        file_name="business_analysis.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.warning(f"⚠️ PDF failed: {str(e)[:50]}")
        
        # Summary
        st.success("✅ Analysis complete!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Agents", "8")
        with col2:
            st.metric("Region", region)
        with col3:
            st.metric("Model", model_all)
        
        # LangSmith link
        if langsmith_enabled:
            st.divider()
            st.info("🔍 [View traces in LangSmith](https://smith.langchain.com/)")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Check:\n- API key valid\n- Rate limits\n- Network\n- API tier for o-series models")
        
        if langsmith_enabled:
            st.info("[Check LangSmith](https://smith.langchain.com/) for details")
        st.stop()
