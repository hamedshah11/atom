import os, sys
from pathlib import Path
import streamlit as st
import time

st.set_page_config(page_title="Business Idea Analyzer Pro", page_icon="🚀", layout="wide")

# ---- Load API keys from Streamlit secrets ----
if "openai_api_key" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    
# Load other settings
if "model_all" in st.secrets:
    os.environ["MODEL_ALL"] = st.secrets["model_all"]
else:
    os.environ["MODEL_ALL"] = os.getenv("MODEL_ALL", "gpt-4-turbo")

# Enable Serper if key exists
if os.getenv("SERPER_API_KEY"):
    os.environ["USE_SERPER"] = "1"
elif "serper_api_key" in st.secrets:
    os.environ["SERPER_API_KEY"] = st.secrets["serper_api_key"]
    os.environ["USE_SERPER"] = "1"

os.environ["VERBOSITY"] = os.getenv("VERBOSITY", "high")
os.environ["DEBUG_MODE"] = os.getenv("DEBUG_MODE", "0")
os.environ["RATE_LIMIT_DELAY"] = os.getenv("RATE_LIMIT_DELAY", "1.5")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.pdf import generate_analysis_pdf

# Custom CSS for better UI
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #00cc88;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚀 Business Idea Analyzer Pro")
    st.caption("AI-powered comprehensive business analysis with real-time market intelligence")
with col2:
    st.image("https://via.placeholder.com/150x50/00cc88/ffffff?text=Pro+Version", width=150)

# Configuration status
with st.expander("⚙️ Configuration Status", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        api_status = "✅ Connected" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
        st.metric("OpenAI API", api_status)
    
    with col2:
        model = os.getenv("MODEL_ALL", "Not set")
        st.metric("AI Model", model)
    
    with col3:
        serper_status = "✅ Enabled" if os.getenv("USE_SERPER") == "1" else "❌ Disabled"
        st.metric("Web Search", serper_status)
    
    with col4:
        delay = os.getenv("RATE_LIMIT_DELAY", "1.5")
        st.metric("API Delay", f"{delay}s")

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not configured!")
    st.info("""
    To configure:
    1. For Streamlit Cloud: Go to ⋮ → Settings → Secrets and add:
       ```
       openai_api_key = "sk-..."
       serper_api_key = "..." # Optional for web search
       ```
    2. For local: Set environment variables or create .env file
    """)
    st.stop()

# Main input section
st.markdown("## 📝 Business Idea Input")

col1, col2 = st.columns([3, 1])

with col1:
    idea = st.text_area(
        "Describe your business idea",
        height=120,
        placeholder="Example: A nursing education institute offering undergraduate and postgraduate programs in healthcare...",
        help="Be specific about your business model, target market, and unique value proposition"
    )

with col2:
    region = st.text_input(
        "Target Region",
        value="Pakistan",
        help="Primary geographic market"
    )
    
    analysis_depth = st.select_slider(
        "Analysis Depth",
        options=["Quick", "Standard", "Comprehensive"],
        value="Comprehensive",
        help="Comprehensive includes web research and deeper analysis"
    )

# Advanced options
with st.expander("🎯 Advanced Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_regulatory = st.checkbox("Include Regulatory Analysis", value=True)
        include_technology = st.checkbox("Include Technology Planning", value=True)
    
    with col2:
        api_delay = st.number_input(
            "API Delay (seconds)",
            min_value=0.5,
            max_value=10.0,
            value=float(os.getenv("RATE_LIMIT_DELAY", "1.5")),
            step=0.5,
            help="Increase if hitting rate limits"
        )
        os.environ["RATE_LIMIT_DELAY"] = str(api_delay)
    
    with col3:
        export_format = st.selectbox(
            "Export Format",
            ["PDF", "Markdown", "Both"],
            help="Choose report export format"
        )

# Analysis button
analyze_button = st.button(
    "🔍 Analyze Business Idea",
    type="primary",
    use_container_width=True,
    disabled=not idea.strip()
)

if analyze_button and idea.strip():
    start_time = time.time()
    
    # Initialize results container
    results = {}
    
    # Progress tracking
    st.markdown("---")
    progress_container = st.container()
    
    with progress_container:
        st.subheader("📊 Analysis Progress")
        
        # Define all agents with their info
        agent_list = [
            ("📋 Strategic Planning", "planner", agents.planner_agent, (idea, region)),
            ("📈 Market Analysis", "market", agents.market_analysis_agent, (idea, region)),
            ("🏢 Competition Analysis", "competition", agents.competition_analysis_agent, (idea, region)),
            ("💰 Financial Feasibility", "financial", agents.financial_feasibility_agent, (idea, region)),
            ("🎯 Go-to-Market Strategy", "gtm", agents.gtm_strategy_agent, (idea, region)),
            ("⚠️ Risk Analysis", "risks", agents.risks_analysis_agent, (idea, region)),
        ]
        
        # Add optional agents
        if include_regulatory:
            agent_list.append(("📜 Regulatory Compliance", "regulatory", agents.regulatory_compliance_agent, (idea, region)))
        if include_technology:
            agent_list.append(("💻 Technology Infrastructure", "technology", agents.technology_infrastructure_agent, (idea, region)))
        
        # Add final agents
        agent_list.extend([
            ("🔍 Critical Review", "critic", None, None),  # Special handling
            ("📊 Final Synthesis", "final_report", None, None)  # Special handling
        ])
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Time estimate
        total_agents = len(agent_list)
        time_estimate = total_agents * 15  # 15 seconds per agent estimate
        st.info(f"⏱️ Estimated time: {time_estimate // 60} minutes {time_estimate % 60} seconds")
        
        # Results display area
        results_container = st.container()
    
    # Run each agent
    for i, (name, key, agent_func, args) in enumerate(agent_list):
        try:
            status_text.text(f"🔄 Running {name}...")
            progress = int((i / total_agents) * 100)
            progress_bar.progress(progress)
            
            # Special handling for critic and synthesizer
            if key == "critic":
                # Gather all analyses for critic
                analyses = {
                    "Market": results.get("market", ""),
                    "Competition": results.get("competition", ""),
                    "Financial": results.get("financial", ""),
                    "GTM": results.get("gtm", ""),
                    "Risks": results.get("risks", ""),
                    "Regulatory": results.get("regulatory", ""),
                    "Technology": results.get("technology", "")
                }
                result = agents.critic_agent(idea, analyses, region)
            
            elif key == "final_report":
                # Gather everything for final synthesis
                analyses = {
                    "Market": results.get("market", ""),
                    "Competition": results.get("competition", ""),
                    "Financial": results.get("financial", ""),
                    "GTM": results.get("gtm", ""),
                    "Risks": results.get("risks", ""),
                    "Regulatory": results.get("regulatory", ""),
                    "Technology": results.get("technology", "")
                }
                critique = results.get("critic", "")
                result = agents.synthesizer_agent(idea, analyses, critique, region)
            
            else:
                # Run normal agent
                result = agent_func(*args)
            
            # Store result
            results[key] = result
            
            # Display result in expandable section
            with results_container:
                with st.expander(f"{name} - ✅ Complete", expanded=(key == "final_report")):
                    if key == "final_report":
                        # Special formatting for final report
                        st.markdown(result)
                    else:
                        # Regular display
                        st.write(result)
            
            # Small delay to show progress
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"❌ Error in {name}: {str(e)}")
            results[key] = f"Error: {str(e)}"
    
    # Complete progress
    progress_bar.progress(100)
    status_text.text("✅ Analysis Complete!")
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary section
    st.markdown("---")
    st.subheader("📊 Analysis Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Analysis Time", f"{total_time:.1f}s")
    with col2:
        successful = sum(1 for v in results.values() if v and not v.startswith("Error:"))
        st.metric("Successful Analyses", f"{successful}/{total_agents}")
    with col3:
        web_search_status = "Used" if os.getenv("USE_SERPER") == "1" else "Not Used"
        st.metric("Web Research", web_search_status)
    with col4:
        word_count = sum(len(str(v).split()) for v in results.values())
        st.metric("Total Word Count", f"{word_count:,}")
    
    # Export section
    st.markdown("---")
    st.subheader("📥 Export Report")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Generate PDF
        if export_format in ["PDF", "Both"]:
            with st.spinner("Generating PDF report..."):
                pdf_bytes = generate_analysis_pdf(
                    idea=idea,
                    plan=results.get('planner', ''),
                    market=results.get('market', ''),
                    competition=results.get('competition', ''),
                    financial=results.get('financial', ''),
                    gtm=results.get('gtm', ''),
                    risks=results.get('risks', ''),
                    critic=results.get('critic', ''),
                    final_report=results.get('final_report', ''),
                    regulatory=results.get('regulatory', ''),
                    technology=results.get('technology', '')
                )
            
            if pdf_bytes:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"business_analysis_{timestamp}.pdf"
                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.error("Failed to generate PDF")
    
    with col2:
        # Generate Markdown
        if export_format in ["Markdown", "Both"]:
            markdown_content = f"# Business Analysis Report\n\n"
            markdown_content += f"**Business Idea**: {idea}\n"
            markdown_content += f"**Target Region**: {region}\n"
            markdown_content += f"**Analysis Date**: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
            
            for name, key, _, _ in agent_list:
                if key in results and results[key]:
                    markdown_content += f"## {name}\n\n"
                    markdown_content += results[key] + "\n\n"
                    markdown_content += "---\n\n"
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"business_analysis_{timestamp}.md"
            st.download_button(
                "📝 Download Markdown",
                data=markdown_content,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )
    
    with col3:
        # Quick actions
        if st.button("📋 Copy Executive Summary", use_container_width=True):
            if 'final_report' in results:
                # Extract executive summary
                exec_summary = results['final_report'].split('##')[1] if '##' in results['final_report'] else results['final_report'][:500]
                st.code(exec_summary, language="markdown")
    
    # Success message
    st.markdown("---")
    st.success(f"""
    ✅ **Analysis completed successfully!**
    
    - Generated {successful} comprehensive analyses
    - Processed {word_count:,} words of insights
    - Completed in {total_time:.1f} seconds
    
    Review the detailed analyses above and download the report for your records.
    """)

# Help section
with st.expander("ℹ️ Help & Best Practices"):
    st.markdown("""
    ### 🎯 Tips for Better Results
    
    **Business Idea Description**:
    - Be specific about your business model
    - Include target customer segments
    - Mention unique value propositions
    - Specify the scale you're targeting
    
    **Web Search Integration** (if enabled):
    - The system will automatically search for:
      - Market size and growth data
      - Competitor information
      - Regulatory requirements
      - Industry trends and insights
    
    **Analysis Depth**:
    - **Quick**: Basic analysis without deep research
    - **Standard**: Balanced analysis with moderate detail
    - **Comprehensive**: Full analysis with web research and maximum detail
    
    ### 🚀 Features
    
    - **10 Specialized AI Agents**: Each focused on a specific aspect
    - **Real-time Web Research**: Current market data (if Serper enabled)
    - **Regional Insights**: Tailored to your target market
    - **Professional Reports**: Export as PDF or Markdown
    - **Actionable Recommendations**: Clear next steps
    
    ### ⚠️ Troubleshooting
    
    - **Rate Limit Errors**: Increase API delay in advanced options
    - **Empty Results**: Check your API key and internet connection
    - **PDF Generation Issues**: Try Markdown export as alternative
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Powered by GPT-4 & LangGraph | Enhanced with Web Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
