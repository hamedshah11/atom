from fpdf import FPDF

def generate_analysis_pdf(idea: str, plan: str, market: str, competition: str, financial: str,
                          gtm: str, risks: str, critic: str, final_report: str,
                          chart_path: str = None, graph_path: str = None) -> bytes:
    """
    Create a PDF report with all the analysis content and visuals.
    Returns PDF data as bytes.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Business Idea Analysis Report", ln=1, align='C')
    pdf.ln(5)
    # Section: Business Idea
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Business Idea:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, idea, ln=1)
    pdf.ln(5)
    # Section: Plan
    if plan:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Analysis Plan:", ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, plan, ln=1)
        pdf.ln(5)
    # Section: Market Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Market Analysis:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, market, ln=1)
    pdf.ln(5)
    # Section: Competition Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Competition Analysis:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, competition, ln=1)
    pdf.ln(5)
    # Section: Financial Feasibility
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Financial Feasibility:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, financial, ln=1)
    pdf.ln(5)
    # Section: Go-to-Market Strategy
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Go-to-Market Strategy:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, gtm, ln=1)
    pdf.ln(5)
    # Section: Risks Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Risks Analysis:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, risks, ln=1)
    pdf.ln(5)
    # Section: Critic Feedback
    if critic:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Critique:", ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 8, critic, ln=1)
        pdf.ln(5)
    # Section: Final Report
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Final Synthesis Report:", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, final_report, ln=1)
    # Add a new page for visualizations if images provided
    if chart_path or graph_path:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Visualizations", ln=1)
        pdf.ln(5)
        pdf.set_font("Arial", '', 12)
        if chart_path:
            try:
                pdf.image(chart_path, w=150)  # adjust width as needed
                pdf.ln(2)
                pdf.cell(0, 10, "Figure: TAM/SAM/SOM Market Size Chart", ln=1)
            except Exception as e:
                pdf.cell(0, 10, f"(Failed to load chart image: {e})", ln=1)
        if graph_path:
            pdf.ln(5)
            try:
                pdf.image(graph_path, w=150)
                pdf.ln(2)
                pdf.cell(0, 10, "Figure: Agent Workflow DAG", ln=1)
            except Exception as e:
                pdf.cell(0, 10, f"(Failed to load graph image: {e})", ln=1)
    # Output PDF to bytes
    try:
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        print(f"Error generating PDF bytes: {e}")
        return None
    return pdf_bytes
