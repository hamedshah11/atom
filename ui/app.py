# ui/app.py
import streamlit as st
import openai
import sys
from pathlib import Path

# Ensure project root is on sys.path (so "backend" and "utils" resolve)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import agents
from utils.visuals import make_market_chart, draw_agent_graph
from utils.pdf import generate_analysis_pdf

# Title and description
st.title("Business Idea Analyzer")
st.write("Enter a business idea to analyze its market, competition, financial feasibility, go-to-market strategy, and risks. This app uses multiple AI agents (Planner, Analysts, Critic, Synthesizer) to evaluate your idea.")

# API Key input if not set in environment (for local usage)
if not openai.api_key:
    api_key = st.text_input("OpenAI API Key (required)", type="password")
    if api_key:
        openai.api_key = api_key

# User input for the business idea
idea_input = st.text_area("Business Idea", placeholder="Describe your business idea here...", height=150)

if st.button("Analyze Idea"):
    if not idea_input.strip():
        st.error("Please enter a business idea to analyze.")
    else:
        # 1. Planner: create analysis plan
        st.subheader("1. Planner Output")
        plan_text = agents.planner_agent(idea_input.strip())
        st.write(plan_text)
        # 2. Market Analysis
        st.subheader("2. Market Analysis")
        market_text = agents.market_analysis_agent(idea_input)
        st.write(market_text)
        # 3. Competition Analysis
        st.subheader("3. Competition Analysis")
        competition_text = agents.competition_analysis_agent(idea_input)
        st.write(competition_text)
        # 4. Financial Feasibility
        st.subheader("4. Financial Feasibility")
        financial_text = agents.financial_feasibility_agent(idea_input)
        st.write(financial_text)
        # 5. Go-to-Market Strategy
        st.subheader("5. Go-to-Market Strategy")
        gtm_text = agents.gtm_strategy_agent(idea_input)
        st.write(gtm_text)
        # 6. Risks Analysis
        st.subheader("6. Risks Analysis")
        risks_text = agents.risks_analysis_agent(idea_input)
        st.write(risks_text)
        # 7. Critic Feedback
        st.subheader("7. Critique (Critic Agent)")
        analyses_dict = {
            "Market": market_text,
            "Competition": competition_text,
            "Financial": financial_text,
            "GTM": gtm_text,
            "Risks": risks_text
        }
        critic_text = agents.critic_agent(idea_input, analyses_dict)
        st.write(critic_text)
        # 8. Final Synthesized Report
        st.subheader("8. Final Synthesized Report")
        final_report_text = agents.synthesizer_agent(idea_input, analyses_dict, critic_text)
        # Display final report with markdown (it may contain structured content)
        st.markdown(final_report_text)
        # Visualizations
        chart_path = make_market_chart(market_text, filename="tam_sam_som_chart.png")
        graph_path = draw_agent_graph(filename="agent_workflow.png")
        if chart_path:
            st.image(chart_path, caption="TAM/SAM/SOM Market Size Chart")
        if graph_path:
            st.image(graph_path, caption="Agent Workflow DAG")
        # PDF Download
        pdf_data = generate_analysis_pdf(
            idea=idea_input.strip(), plan=plan_text,
            market=market_text, competition=competition_text,
            financial=financial_text, gtm=gtm_text,
            risks=risks_text, critic=critic_text,
            final_report=final_report_text,
            chart_path=chart_path, graph_path=graph_path
        )
        if pdf_data:
            st.download_button(label="Download Full Analysis as PDF", data=pdf_data, file_name="Business_Idea_Analysis.pdf", mime="application/pdf")
        else:
            st.error("Failed to generate PDF.")
