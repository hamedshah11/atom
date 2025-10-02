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

# Get current model dynamically
current_model = os.getenv("MODEL_ALL", "gpt-4o-mini")

# Status bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    model_emoji = {
        "gpt-4o": "🌟",
        "gpt-4o-mini": "⚡",
        "o1": "🧠",
        "o3-mini": "🤔",
        "o1-preview": "⚠️",
        "o1-mini": "⚠️",
        "gpt-4-turbo": "❌",
        "gpt-3.5-turbo": "❌"
    }
    emoji = model_emoji.get(current_model, "🤖")
    st.info(f"{emoji} Active Model: **{current_model}**")
    
    if current_model in ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
        st.error("❌ This model is deprecated!")
    elif current_model in ["o1-preview", "o1-mini"]:
        st.warning("⚠️ Legacy model!")

with col2:
    if langsmith_enabled:
        st.success("🔍 LangSmith: ON")
    else:
        st.info("🔍 LangSmith: OFF")

with col3:
    st.info(f"🔎 Search: ON")

# Debug sidebar
with st.sidebar:
    # Show which model is ACTUALLY being used
    st.info(f"**🎯 Current Model:**\n{current_model}")
    
    with st.expander("🐛 Debug Info", expanded=False):
        st.write("**Environment:**")
        st.write(f"MODEL_ALL: {os.getenv('MODEL_ALL', 'not set')}")
        st.write(f"API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
        
        st.divider()
        st.write("**LangSmith:**")
        st.write(f"API Key: {'✅ Set' if os.getenv('LANGCHAIN_API_KEY') else '❌ Missing'}")
        st.write(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
        st.write(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
        
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
        
        # Model info
        models_info = {
            "gpt-4o-mini": {
                "emoji": "⚡", 
                "cost": "$0.02", 
                "speed": "★★★★★", 
                "quality": "★★★★☆", 
                "desc": "Best value",
                "note": "No tier requirement"
            },
            "gpt-4o": {
                "emoji": "🌟", 
                "cost": "$0.08", 
                "speed": "★★★★☆", 
                "quality": "★★★★★", 
                "desc": "Best quality",
                "note": "No tier requirement"
            },
            "o1": {
                "emoji": "🧠", 
                "cost": "$0.50", 
                "speed": "★★☆☆☆", 
                "quality": "★★★★★", 
                "desc": "Advanced reasoning",
                "note": "⚠️ Requires API Tier 1+ ($5 spend)"
            },
            "o3-mini": {
                "emoji": "🤔", 
                "cost": "$0.15", 
                "speed": "★★★☆☆", 
                "quality": "★★★★☆", 
                "desc": "Efficient reasoning",
                "note": "⚠️ Requires API Tier 1+ ($5 spend)"
            }
        }
        
        # Show currently ACTIVE model
        if current_model in models_info:
            current_info = models_info[current_model]
            st.success(f"**🎯 Active Model:** {current_info['emoji']} **{current_model}** — {current_info['cost']}/analysis")
        else:
            st.warning(f"⚠️ Active: {current_model} (unknown)")
        
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
                    st.caption(f"{info['note']}")
                    
                    if st.button(f"Use {model}", key=f"select_{model}", 
                               disabled=(model == current_model),
                               use_container_width=True):
                        os.environ["MODEL_ALL"] = model
                        st.success(f"✅ Switched to {model}!")
                        st.info("Model will be used in next analysis")
                        # Force a small delay to ensure env var is set
                        import time
                        time.sleep(0.1)
                        st.rerun()
        
        st.divider()
        
        with st.expander("⚠️ Legacy Models", expanded=False):
            st.warning("**Replaced models:**\n"
                      "- ❌ o1-preview → Use **o1**\n"
                      "- ❌ o1-mini → Use **o3-mini**\n"
                      "- ❌ gpt-4-turbo → Use **gpt-4o**")
    
    with tab3:
        st.subheader("LangSmith Observability")
        
        if not langsmith_enabled:
            st.info("Enable to track executions")
            
            key = st.text_input("LangSmith API Key", type="password")
            proj = st.text_input("Project", value="business-idea-analyzer")
            
            if st.button("Enable LangSmith", type="primary"):
                if key:
                    os.environ["LANGCHAIN_API_KEY"] = key
                    os.environ["LANGCHAIN_TRACING_V2"] = "true"
                    os.environ["LANGCHAIN_PROJECT"] = proj
                    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                    st.success("✅ Enabled!")
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
    
    # Show which model will be used
    active_model = os.getenv("MODEL_ALL", "gpt-4o-mini")
    st.info(f"🤖 Using model: **{active_model}**")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Planner
        status_text.text(f"🔄 Planner Agent ({active_model})...")
        with st.spinner("Planning..."):
            plan = agents.planner_agent(idea)
        st.subheader("1) Planner")
        st.write(plan)
        progress_bar.progress(12)
        
        # Market
        status_text.text(f"🔄 Market Analysis ({active_model})...")
        with st.spinner("Analyzing market..."):
            market = agents.market_analysis_agent(idea, region)
        st.subheader("2) Market Analysis")
        st.write(market)
        progress_bar.progress(25)
        
        # Competition
        status_text.text(f"🔄 Competition Analysis ({active_model})...")
        with st.spinner("Analyzing competition..."):
            competition = agents.competition_analysis_agent(idea, region)
        st.subheader("3) Competition")
        st.write(competition)
        progress_bar.progress(38)
        
        # Financial
        status_text.text(f"🔄 Financial Analysis ({active_model})...")
        with st.spinner("Financial modeling..."):
            financial = agents.financial_feasibility_agent(idea, region)
        st.subheader("4) Financial Feasibility")
        st.write(financial)
        progress_bar.progress(50)
        
        # GTM
        status_text.text(f"🔄 Go-to-Market ({active_model})...")
        with st.spinner("GTM strategy..."):
            gtm = agents.gtm_strategy_agent(idea, region)
        st.subheader("5) Go-to-Market")
        st.write(gtm)
        progress_bar.progress(62)
        
        # Risks
        status_text.text(f"🔄 Risk Analysis ({active_model})...")
        with st.spinner("Identifying risks..."):
            risks = agents.risks_analysis_agent(idea, region)
        st.subheader("6) Risks")
        st.write(risks)
        progress_bar.progress(75)
        
        # Critique
        status_text.text(f"🔄 Critique ({active_model})...")
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
        status_text.text(f"🔄 Synthesis ({active_model})...")
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
            st.metric("Model Used", active_model)
        
        # LangSmith link
        if langsmith_enabled:
            st.divider()
            st.info("🔍 [View traces in LangSmith](https://smith.langchain.com/)")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Check:\n- API key valid\n- Rate limits\n- Network\n- API tier for o-series")
        
        if langsmith_enabled:
            st.info("[Check LangSmith](https://smith.langchain.com/)")
        st.stop()
