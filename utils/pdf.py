from fpdf import FPDF

def generate_analysis_pdf(
    idea: str, plan: str, market: str, competition: str, financial: str,
    gtm: str, risks: str, critic: str, final_report: str,
    chart_path: str = None, graph_path: str = None
) -> bytes | None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    def h(txt): pdf.set_font("Arial", "B", 14); pdf.cell(0, 9, txt, ln=1)
    def p(txt): pdf.set_font("Arial", "", 12); pdf.multi_cell(0, 6, txt); pdf.ln(2)

    pdf.set_font("Arial", "B", 16); pdf.cell(0, 10, "Business Idea Analysis Report", ln=1)
    pdf.ln(2)

    h("Business Idea");        p(idea)
    if plan: h("Analysis Plan"); p(plan)
    h("Market Analysis");      p(market)
    h("Competition Analysis"); p(competition)
    h("Financial Feasibility");p(financial)
    h("Go-to-Market Strategy");p(gtm)
    h("Risks Analysis");       p(risks)
    if critic: h("Critique");  p(critic)
    h("Final Synthesis Report"); p(final_report)

    if chart_path or graph_path:
        pdf.add_page()
        h("Visualizations")
        if chart_path:
            try:
                pdf.image(chart_path, w=170); pdf.ln(4); p("Figure: TAM/SAM/SOM")
            except Exception as e:
                p(f"(Failed to load chart image: {e})")
        if graph_path:
            try:
                pdf.image(graph_path, w=170); pdf.ln(4); p("Figure: Agent Workflow")
            except Exception as e:
                p(f"(Failed to load graph image: {e})")

    try:
        return pdf.output(dest="S").encode("latin-1")
    except Exception:
        return None
