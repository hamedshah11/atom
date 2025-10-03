import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer - O3-mini", page_icon="🧠", layout="wide")

# Configure Serper
os.environ["SERPER_API_KEY"] = "e62e1e8919e57b754e3acd05b2d2bb570effb93e"
os.environ["USE_SERPER"] = "1"

# ---- Set O3-mini as default model ----
os.environ["MODEL_ALL"] = "o3-mini"

# ---- Secrets → env BEFORE imports ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
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
    os.environ["LANGCHAIN_PROJECT"] = "business-idea-analyzer-o3mini"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

st.title("🧠 Business Idea Analyzer — O3-mini Reasoning")
st.caption("Powered by OpenAI O3-mini: Advanced reasoning model with LangChain compatibility")

# Check configurations
langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "business-idea-analyzer-o3mini")

# Status bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.success("🧠 Model: **O3-mini** (Reasoning Engine)")

with col2:
    if langsmith_enabled:
        st.success("🔍 LangSmith: ON")
    else:
        st.info("🔍 LangSmith: OFF")

with col3:
    st.info(f"🔎 Search: ON")

# Sidebar configuration
with st.sidebar:
    st.header("🧠 O3-mini Configuration")
    
    # Model info
    st.info(
        "**O3-mini Features:**\n"
        "✅ Advanced reasoning\n"
        "✅ Streaming support\n"
        "✅ Tool calling\n"
        "✅ System messages\n"
        "⚠️ Requires Tier 1+ ($5 spend)"
    )
    
    with st.expander("🔧 System Status", expanded=False):
        st.write("**Environment:**")
        st.write(f"Model: o3-mini")
        st.write(f"Fallback: gpt-4o")
        st.write(f"API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
        
        st.divider()
        st.write("**LangSmith:**")
        st.write(f"API Key: {'✅ Set' if os.getenv('LANGCHAIN_API_KEY') else '❌ Missing'}")
        st.write(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
        st.write(f"Project: {os.getenv('LANGCHAIN_PROJECT')}")
        
        st.divider()
        
        if st.button("🧪 Test API"):
            try:
                from openai import OpenAI
                client = OpenAI()
                with st.spinner("Testing o3-mini..."):
                    try:
                        # Try o3-mini first
                        response = client.chat.completions.create(
                            model="o3-mini",
                            messages=[{"role": "user", "content": "Say 'O3-mini working'"}],
                            max_completion_tokens=20,
                            reasoning_effort="low"
                        )
                        st.success(f"✅ O3-mini: {response.choices[0].message.content}")
                    except Exception as e:
                        if "tier" in str(e).lower() or "quota" in str(e).lower():
                            st.warning("⚠️ O3-mini requires Tier 1+")
                            # Test fallback
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": "Say 'GPT-4o fallback active'"}],
                                max_tokens=20
                            )
                            st.success(f"✅ Fallback: {response.choices[0].message.content}")
                        else:
                            st.error(f"❌ {str(e)[:100]}")
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)[:100]}")

# Configuration panel
with st.expander("⚙️ Configuration", expanded=False):
    tab1, tab2, tab3 = st.tabs(["OpenAI Setup", "Model Info", "LangSmith"])
    
    with tab1:
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ API Key Required")
            key = st.text_input("OpenAI API Key", type="password", 
                              help="Required for O3-mini access")
            if key:
                os.environ["OPENAI_API_KEY"] = key
                st.success("✅ Set for session")
                st.rerun()
        else:
            st.success("✅ OpenAI configured")
            
            # Check API tier
            st.info(
                "**Important:** O3-mini requires API Tier 1+ ($5+ spent on OpenAI API).\n\n"
                "If you don't have Tier 1 access, the system will automatically fall back to GPT-4o."
            )
    
    with tab2:
        st.subheader("🧠 O3-mini Reasoning Model")
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model", "O3-mini")
            st.metric("Cost/Analysis", "~$0.15")
            st.metric("Speed", "Moderate")
        
        with col2:
            st.metric("Reasoning", "Advanced")
            st.metric("Context", "128K tokens")
            st.metric("LangChain", "✅ Compatible")
        
        st.divider()
        
        st.write("### Key Advantages over O1:")
        st.write(
            "- ✅ **Streaming support** (o1 doesn't support)\n"
            "- ✅ **Tool calling** (o1 doesn't support)\n"
            "- ✅ **System messages** (o1 doesn't support)\n"
            "- ✅ **Structured outputs** (o1 doesn't support)\n"
            "- ✅ **Works with LangChain** (o1 has issues)"
        )
        
        st.divider()
        
        st.write("### Automatic Fallback")
        st.info(
            "If O3-mini is unavailable (tier restriction), the system automatically "
            "falls back to **GPT-4o** to ensure uninterrupted service."
        )
        
        st.divider()
        st.caption("Note: O4-mini doesn't exist yet as of October 2025")
    
    with tab3:
        st.subheader("LangSmith Observability")
        
        if not langsmith_enabled:
            st.info("Enable to track O3-mini reasoning traces")
            
            key = st.text_input("LangSmith API Key", type="password")
            proj = st.text_input("Project", value="business-idea-analyzer-o3mini")
            
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
st.header("📊 Business Idea Analysis")
idea = st.text_area("Your business idea", height=140, 
                    placeholder="Describe your innovative business concept for O3-mini to analyze...")
region = st.text_input("Target region", value="Pakistan")

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button("🧠 Analyze with O3-mini", type="primary", use_container_width=True)
with col2:
    st.metric("Reasoning", "Advanced")

if run:
    if not idea.strip():
        st.error("Please enter a business idea")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Configure OpenAI API key first")
        st.stop()
    
    st.info("🧠 Analyzing with O3-mini reasoning model...")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analysis stages
        stages = [
            ("🎯 Strategic Planning", agents.planner_agent, idea, 12),
            ("📊 Market Analysis", lambda i: agents.market_analysis_agent(i, region), idea, 25),
            ("🏆 Competition Analysis", lambda i: agents.competition_analysis_agent(i, region), idea, 38),
            ("💰 Financial Modeling", lambda i: agents.financial_feasibility_agent(i, region), idea, 50),
            ("🚀 Go-to-Market Strategy", lambda i: agents.gtm_strategy_agent(i, region), idea, 62),
            ("⚠️ Risk Assessment", lambda i: agents.risks_analysis_agent(i, region), idea, 75),
        ]
        
        results = {}
        model_used = "o3-mini"  # Track which model was actually used
        
        for idx, (stage_name, agent_func, input_data, progress) in enumerate(stages):
            status_text.text(f"🔄 {stage_name} (O3-mini reasoning)...")
            with st.spinner(f"Processing {stage_name}..."):
                result = agent_func(input_data)
                
                # Check if we got a fallback message
                if "Tier restriction" in result and idx == 0:
                    model_used = "gpt-4o (fallback)"
                    st.warning("⚠️ O3-mini unavailable (Tier 1+ required). Using GPT-4o fallback.")
                
                results[stage_name] = result
            
            st.subheader(stage_name)
            st.write(result)
            progress_bar.progress(progress)
        
        # Critique
        status_text.text(f"🔄 Critical Analysis...")
        analyses = {
            "Market": results["📊 Market Analysis"],
            "Competition": results["🏆 Competition Analysis"],
            "Financial": results["💰 Financial Modeling"],
            "GTM": results["🚀 Go-to-Market Strategy"],
            "Risks": results["⚠️ Risk Assessment"]
        }
        
        with st.spinner("Performing critical review..."):
            critic = agents.critic_agent(idea, analyses, region)
        st.subheader("🔍 Critical Review")
        st.write(critic)
        progress_bar.progress(88)
        
        # Final synthesis
        status_text.text(f"🔄 Executive Synthesis...")
        with st.spinner("Generating executive report..."):
            final_md = agents.synthesizer_agent(idea, analyses, critic, region)
        st.subheader("📋 Executive Report")
        st.markdown(final_md)
        progress_bar.progress(100)
        
        status_text.text("✅ Analysis Complete!")
        
        # PDF Generation
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_analysis_pdf(
                    idea, 
                    results["🎯 Strategic Planning"],
                    results["📊 Market Analysis"],
                    results["🏆 Competition Analysis"],
                    results["💰 Financial Modeling"],
                    results["🚀 Go-to-Market Strategy"],
                    results["⚠️ Risk Assessment"],
                    critic, 
                    final_md
                )
                if pdf_bytes:
                    st.download_button(
                        "📄 Download Executive Report (PDF)",
                        data=pdf_bytes,
                        file_name="o3mini_business_analysis.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"⚠️ PDF generation issue: {str(e)[:50]}")
        
        # Summary metrics
        st.success("✅ Analysis Complete!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stages", "8 Complete")
        with col2:
            st.metric("Region", region)
        with col3:
            st.metric("Model", model_used)
        with col4:
            st.metric("Reasoning", "Advanced")
        
        # LangSmith link
        if langsmith_enabled:
            st.divider()
            st.info("🔍 [View O3-mini reasoning traces in LangSmith](https://smith.langchain.com/)")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        error_msg = str(e)
        
        if "tier" in error_msg.lower() or "quota" in error_msg.lower():
            st.warning(
                "**O3-mini Access Issue**\n\n"
                "O3-mini requires API Tier 1+ ($5+ spent on OpenAI API).\n\n"
                "**Solutions:**\n"
                "1. The system should have automatically fallen back to GPT-4o\n"
                "2. Check your OpenAI API tier at platform.openai.com\n"
                "3. Verify your API key is correct"
            )
        elif "rate limit" in error_msg.lower():
            st.info("Rate limit reached. Please wait a moment and try again.")
        else:
            st.info(
                "**Troubleshooting:**\n"
                "- Verify API key is valid\n"
                "- Check network connection\n"
                "- Ensure you have API Tier 1+ for O3-mini"
            )
        
        if langsmith_enabled:
            st.info("[Check error details in LangSmith](https://smith.langchain.com/)")
        st.stop()
