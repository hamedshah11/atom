import os, sys
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Business Idea Analyzer", page_icon="🧠", layout="wide")

# ---- CRITICAL: Set model BEFORE any imports ----
# Use GPT-4o for reliability (change to o3-mini if you have Tier 1+ access)
os.environ["MODEL_ALL"] = "gpt-4o"

# Configure Serper
os.environ["USE_SERPER"] = "1"
if "serper_api_key" not in st.secrets:
    os.environ["SERPER_API_KEY"] = "e62e1e8919e57b754e3acd05b2d2bb570effb93e"

# ---- Secrets → env BEFORE imports ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
else:
    # Check if it's in regular env
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY not found in Streamlit secrets or environment variables!")
        st.info("Add it to `.streamlit/secrets.toml` or set as environment variable")
        st.stop()

if "verbosity" in st.secrets:
    os.environ["VERBOSITY"] = st.secrets["verbosity"]
    
if "serper_api_key" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["serper_api_key"]

# Rate limiting from secrets or default
if "rate_limit_delay" in st.secrets:
    os.environ["RATE_LIMIT_DELAY"] = str(st.secrets["rate_limit_delay"])
else:
    os.environ["RATE_LIMIT_DELAY"] = "2.5"

# LangSmith configuration
if "langchain_api_key" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["langchain_api_key"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("langchain_tracing_v2", "true")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    
if "langchain_project" in st.secrets:
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["langchain_project"]
else:
    os.environ["LANGCHAIN_PROJECT"] = "business-idea-analyzer"

# Debug mode
if "debug_mode" in st.secrets:
    os.environ["DEBUG_MODE"] = str(st.secrets["debug_mode"])

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

# Get current configurations
current_model = os.getenv("MODEL_ALL", "gpt-4o")
langsmith_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
langsmith_project = os.getenv("LANGCHAIN_PROJECT", "business-idea-analyzer")
rate_limit = os.getenv("RATE_LIMIT_DELAY", "2.5")

st.title("🧠 Business Idea Analyzer")
st.caption(f"Powered by OpenAI {current_model.upper()}")

# Status bar
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    if current_model == "o3-mini":
        st.success(f"🧠 Model: **{current_model}** (Reasoning)")
    else:
        st.success(f"🤖 Model: **{current_model}**")

with col2:
    if langsmith_enabled:
        st.success("🔍 LangSmith: ON")
    else:
        st.info("🔍 LangSmith: OFF")

with col3:
    st.info(f"🔎 Search: ON")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "Model",
        ["gpt-4o", "o3-mini", "gpt-4o-mini"],
        index=0 if current_model == "gpt-4o" else (1 if current_model == "o3-mini" else 2),
        help="GPT-4o: Most reliable. O3-mini: Advanced reasoning (requires Tier 1+)"
    )
    
    if model_choice != current_model:
        os.environ["MODEL_ALL"] = model_choice
        st.success(f"✅ Switched to {model_choice}")
        st.rerun()
    
    # Rate limit configuration
    st.divider()
    st.subheader("Performance")
    
    new_rate_limit = st.slider(
        "Rate Limit Delay (seconds)",
        min_value=1.0,
        max_value=5.0,
        value=float(rate_limit),
        step=0.5,
        help="Higher = slower but more reliable"
    )
    
    if new_rate_limit != float(rate_limit):
        os.environ["RATE_LIMIT_DELAY"] = str(new_rate_limit)
        st.info(f"Updated to {new_rate_limit}s")
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        st.write("**Environment:**")
        st.write(f"Model: {current_model}")
        st.write(f"API Key: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing'}")
        st.write(f"Rate Limit: {rate_limit}s")
        st.write(f"Serper: {'✅' if os.getenv('SERPER_API_KEY') else '❌'}")
        
        st.divider()
        st.write("**LangSmith:**")
        st.write(f"Enabled: {langsmith_enabled}")
        if langsmith_enabled:
            st.write(f"Project: {langsmith_project}")
        
        st.divider()
        
        if st.button("🧪 Test API Connection"):
            try:
                from openai import OpenAI
                client = OpenAI()
                
                with st.spinner(f"Testing {current_model}..."):
                    try:
                        # Test current model
                        if current_model == "o3-mini":
                            response = client.chat.completions.create(
                                model="o3-mini",
                                messages=[{"role": "user", "content": "Say 'Working'"}],
                                max_completion_tokens=10,
                                reasoning_effort="low"
                            )
                        else:
                            response = client.chat.completions.create(
                                model=current_model,
                                messages=[{"role": "user", "content": "Say 'Working'"}],
                                max_tokens=10
                            )
                        
                        result = response.choices[0].message.content
                        st.success(f"✅ {current_model}: {result}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "tier" in error_msg.lower() or "quota" in error_msg.lower():
                            st.warning(f"⚠️ {current_model} requires higher API tier")
                            st.info("💡 Switch to gpt-4o in the model dropdown above")
                        else:
                            st.error(f"❌ {error_msg[:150]}")
                            
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)[:150]}")

    # Model info
    with st.expander("ℹ️ Model Information"):
        if current_model == "o3-mini":
            st.info(
                "**O3-mini Features:**\n"
                "✅ Advanced reasoning\n"
                "✅ Tool calling support\n"
                "⚠️ Requires API Tier 1+ ($5 spent)\n\n"
                "**Fallback:** Automatically uses GPT-4o if unavailable"
            )
        elif current_model == "gpt-4o":
            st.success(
                "**GPT-4o Features:**\n"
                "✅ Fast and reliable\n"
                "✅ Works with all tiers\n"
                "✅ Excellent reasoning\n"
                "✅ No special requirements"
            )
        else:
            st.info(
                "**GPT-4o-mini Features:**\n"
                "✅ Fastest response\n"
                "✅ Most cost-effective\n"
                "⚠️ Less advanced reasoning"
            )
    
    # LangSmith toggle
    with st.expander("🔍 LangSmith Observability"):
        if langsmith_enabled:
            st.success("✅ LangSmith tracking enabled")
            st.write(f"**Project:** {langsmith_project}")
            st.write("[View Dashboard →](https://smith.langchain.com/)")
            
            if st.button("Disable LangSmith"):
                os.environ["LANGCHAIN_TRACING_V2"] = "false"
                st.rerun()
        else:
            st.info("Enable to track AI reasoning traces")
            
            with st.form("langsmith_form"):
                key = st.text_input("LangSmith API Key", type="password")
                proj = st.text_input("Project Name", value="business-idea-analyzer")
                
                if st.form_submit_button("Enable LangSmith"):
                    if key:
                        os.environ["LANGCHAIN_API_KEY"] = key
                        os.environ["LANGCHAIN_TRACING_V2"] = "true"
                        os.environ["LANGCHAIN_PROJECT"] = proj
                        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                        st.success("✅ Enabled!")
                        st.rerun()
                    else:
                        st.error("Enter API key")

# Main input
st.header("📊 Business Idea Analysis")

idea = st.text_area(
    "Your business idea",
    height=140,
    placeholder="Describe your business concept in detail...",
    help="Be specific about what you want to build, who it's for, and what problem it solves"
)

col1, col2 = st.columns([3, 1])
with col1:
    region = st.text_input(
        "Target region",
        value="Pakistan",
        help="The geographic market you're targeting"
    )
with col2:
    st.metric("Model", current_model.upper())

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button("🚀 Analyze Business Idea", type="primary", use_container_width=True)
with col2:
    if st.button("🧹 Clear", use_container_width=True):
        st.rerun()

if run:
    if not idea.strip():
        st.error("⚠️ Please enter a business idea")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not configured")
        st.stop()
    
    # Analysis info
    st.info(f"🧠 Analyzing with {current_model.upper()}... (Rate limit: {rate_limit}s between calls)")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analysis stages with proper progress tracking
        stages = [
            ("🎯 Strategic Planning", agents.planner_agent, idea, 12),
            ("📊 Market Analysis", lambda i: agents.market_analysis_agent(i, region), idea, 25),
            ("🏆 Competition Analysis", lambda i: agents.competition_analysis_agent(i, region), idea, 38),
            ("💰 Financial Modeling", lambda i: agents.financial_feasibility_agent(i, region), idea, 50),
            ("🚀 Go-to-Market Strategy", lambda i: agents.gtm_strategy_agent(i, region), idea, 62),
            ("⚠️ Risk Assessment", lambda i: agents.risks_analysis_agent(i, region), idea, 75),
        ]
        
        results = {}
        error_count = 0
        
        for idx, (stage_name, agent_func, input_data, progress) in enumerate(stages):
            status_text.text(f"🔄 {stage_name}...")
            
            with st.spinner(f"Processing {stage_name}..."):
                try:
                    result = agent_func(input_data)
                    
                    # Check if result is an error
                    if result.startswith("⚠️"):
                        error_count += 1
                        st.warning(f"{stage_name}: {result}")
                    
                    results[stage_name] = result
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"⚠️ Error in {stage_name}: {str(e)[:150]}"
                    st.error(error_msg)
                    results[stage_name] = error_msg
            
            # Display result
            with st.expander(f"{stage_name}", expanded=not result.startswith("⚠️")):
                if result.startswith("⚠️"):
                    st.error(result)
                else:
                    st.write(result)
            
            progress_bar.progress(progress)
        
        # Check if too many errors
        if error_count >= 4:
            st.error(
                "⚠️ **Too many API errors detected**\n\n"
                "**Possible solutions:**\n"
                "1. Check your API key is valid\n"
                "2. Increase rate limit delay in sidebar\n"
                f"3. Switch to a different model (currently: {current_model})\n"
                "4. Wait a few minutes and try again"
            )
            st.stop()
        
        # Critique stage
        status_text.text("🔄 Critical Analysis...")
        analyses = {
            "Market": results["📊 Market Analysis"],
            "Competition": results["🏆 Competition Analysis"],
            "Financial": results["💰 Financial Modeling"],
            "GTM": results["🚀 Go-to-Market Strategy"],
            "Risks": results["⚠️ Risk Assessment"]
        }
        
        with st.spinner("Performing critical review..."):
            try:
                critic = agents.critic_agent(idea, analyses, region)
                if not critic.startswith("⚠️"):
                    with st.expander("🔍 Critical Review", expanded=True):
                        st.write(critic)
                else:
                    st.warning(f"Critical Review: {critic}")
            except Exception as e:
                critic = f"⚠️ Critical review failed: {str(e)[:150]}"
                st.warning(critic)
        
        progress_bar.progress(88)
        
        # Final synthesis
        status_text.text("🔄 Executive Synthesis...")
        with st.spinner("Generating executive report..."):
            try:
                final_md = agents.synthesizer_agent(idea, analyses, critic, region)
                
                if not final_md.startswith("⚠️"):
                    st.divider()
                    st.subheader("📋 Executive Report")
                    st.markdown(final_md)
                else:
                    st.warning(f"Executive Report: {final_md}")
                    
            except Exception as e:
                final_md = f"⚠️ Synthesis failed: {str(e)[:150]}"
                st.error(final_md)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis Complete!")
        
        # PDF Generation
        if error_count < 3:  # Only generate PDF if most stages succeeded
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_bytes = generate_analysis_pdf(
                        idea,
                        results.get("🎯 Strategic Planning", ""),
                        results.get("📊 Market Analysis", ""),
                        results.get("🏆 Competition Analysis", ""),
                        results.get("💰 Financial Modeling", ""),
                        results.get("🚀 Go-to-Market Strategy", ""),
                        results.get("⚠️ Risk Assessment", ""),
                        critic,
                        final_md
                    )
                    
                    if pdf_bytes:
                        st.download_button(
                            "📄 Download Executive Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"business_analysis_{region.lower().replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.warning(f"⚠️ PDF generation issue: {str(e)[:100]}")
        
        # Summary metrics
        st.divider()
        st.success("✅ Analysis Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Stages", f"{6 - error_count}/6 Complete")
        with col2:
            st.metric("Region", region)
        with col3:
            st.metric("Model", current_model.upper())
        with col4:
            if error_count == 0:
                st.metric("Status", "✅ Success")
            elif error_count < 3:
                st.metric("Status", "⚠️ Partial")
            else:
                st.metric("Status", "❌ Errors")
        
        # LangSmith link
        if langsmith_enabled:
            st.info(f"🔍 [View detailed traces in LangSmith](https://smith.langchain.com/o/ecommerce-decd3faa/projects/p/{langsmith_project})")

    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        error_msg = str(e).lower()
        
        # Provide helpful troubleshooting
        if "authentication" in error_msg or "api_key" in error_msg:
            st.error(
                "**API Authentication Failed**\n\n"
                "Your OpenAI API key is invalid or not set correctly.\n\n"
                "**Solutions:**\n"
                "1. Check your API key in `.streamlit/secrets.toml`\n"
                "2. Verify the key at https://platform.openai.com/api-keys\n"
                "3. Make sure there are no extra spaces or quotes"
            )
        elif "tier" in error_msg or "quota" in error_msg:
            st.warning(
                f"**{current_model.upper()} Access Issue**\n\n"
                "The selected model is not available for your API tier.\n\n"
                "**Solutions:**\n"
                "1. Switch to GPT-4o using the model dropdown in the sidebar\n"
                "2. Check your API tier at https://platform.openai.com/settings/organization/limits\n"
                "3. For O3-mini, you need Tier 1+ ($5+ spent on OpenAI API)"
            )
        elif "rate limit" in error_msg:
            st.info(
                "**Rate Limit Exceeded**\n\n"
                "**Solutions:**\n"
                "1. Increase 'Rate Limit Delay' in the sidebar (try 3.5s)\n"
                "2. Wait a few minutes before trying again\n"
                "3. Check your rate limits at https://platform.openai.com/settings/organization/limits"
            )
        elif "timeout" in error_msg:
            st.info(
                "**Request Timeout**\n\n"
                "**Solutions:**\n"
                "1. Check your internet connection\n"
                "2. Try again in a moment\n"
                "3. If persistent, switch to a faster model (gpt-4o-mini)"
            )
        else:
            st.info(
                "**Troubleshooting:**\n"
                "1. Verify API key is valid and has credits\n"
                "2. Check network connection\n"
                "3. Try a different model in the sidebar\n"
                "4. Increase rate limit delay\n"
                "5. Enable debug mode to see detailed logs"
            )
        
        if langsmith_enabled:
            st.info("🔍 [Check error details in LangSmith](https://smith.langchain.com/)")
        
        # Show detailed error in expander
        with st.expander("🐛 Detailed Error Information"):
            st.code(str(e))
        
        st.stop()
