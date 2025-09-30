from fpdf import FPDF
import re
from datetime import datetime

class BusinessReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(20, 20, 20)
        
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'I', 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Business Analysis Report', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')
        self.ln(15)
        
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%B %d, %Y")}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add a chapter title"""
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, 0, 1)
        self.ln(4)
        
    def section_title(self, title):
        """Add a section title"""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, title, 0, 1)
        self.ln(2)
        
    def add_text(self, text, style='normal'):
        """Add formatted text"""
        if style == 'normal':
            self.set_font('Arial', '', 11)
            self.set_text_color(0, 0, 0)
        elif style == 'bold':
            self.set_font('Arial', 'B', 11)
            self.set_text_color(0, 0, 0)
        elif style == 'italic':
            self.set_font('Arial', 'I', 11)
            self.set_text_color(64, 64, 64)
        
        # Handle long text by breaking into chunks
        if len(text) > 1000:
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            for chunk in chunks:
                self.multi_cell(0, 6, chunk, 0, 'J')
        else:
            self.multi_cell(0, 6, text, 0, 'J')
        self.ln(2)
    
    def add_bullet_points(self, points):
        """Add bullet points"""
        self.set_font('Arial', '', 11)
        for point in points:
            self.cell(10, 6, chr(149), 0, 0)  # Bullet character
            self.multi_cell(0, 6, point, 0, 'J')
    
    def add_table_row(self, data, widths=None, header=False):
        """Add a table row"""
        if header:
            self.set_font('Arial', 'B', 10)
            self.set_fill_color(230, 230, 230)
        else:
            self.set_font('Arial', '', 10)
            self.set_fill_color(255, 255, 255)
        
        if not widths:
            widths = [self.w / len(data) - 10 for _ in data]
        
        for i, item in enumerate(data):
            self.cell(widths[i], 7, str(item)[:50], 1, 0, 'L', header)
        self.ln()

def clean_text(text):
    """Clean text for PDF generation"""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove markdown formatting but keep structure
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'`(.*?)`', r'\1', text)        # Code
    
    # Fix encoding issues
    text = text.encode('latin-1', 'ignore').decode('latin-1')
    
    return text.strip()

def extract_sections(text):
    """Extract sections from markdown text"""
    sections = []
    current_section = {'title': 'Introduction', 'content': ''}
    
    lines = text.split('\n')
    for line in lines:
        if line.startswith('##') and not line.startswith('###'):
            if current_section['content']:
                sections.append(current_section)
            title = line.replace('##', '').strip()
            current_section = {'title': title, 'content': ''}
        else:
            current_section['content'] += line + '\n'
    
    if current_section['content']:
        sections.append(current_section)
    
    return sections

def generate_analysis_pdf(
    idea: str, plan: str, market: str, competition: str, financial: str,
    gtm: str, risks: str, critic: str, final_report: str,
    regulatory: str = "", technology: str = ""
) -> bytes | None:
    """Generate comprehensive PDF report"""
    try:
        pdf = BusinessReportPDF()
        
        # Title Page
        pdf.add_page()
        pdf.ln(40)
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 15, 'Business Idea Analysis Report', 0, 1, 'C')
        pdf.ln(10)
        pdf.set_font('Arial', 'I', 16)
        pdf.cell(0, 10, idea[:100], 0, 1, 'C')
        pdf.ln(20)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%B %d, %Y")}', 0, 1, 'C')
        
        # Executive Summary (from final report)
        if final_report:
            pdf.add_page()
            pdf.chapter_title('Executive Summary')
            
            # Extract executive summary from final report
            exec_sections = extract_sections(final_report)
            for section in exec_sections[:2]:  # First two sections usually contain summary
                if section['content']:
                    pdf.section_title(section['title'])
                    pdf.add_text(clean_text(section['content']))
        
        # Table of Contents
        pdf.add_page()
        pdf.chapter_title('Table of Contents')
        toc_items = [
            ('1. Strategic Planning', pdf.page_no() + 1),
            ('2. Market Analysis', pdf.page_no() + 2),
            ('3. Competition Analysis', pdf.page_no() + 3),
            ('4. Financial Feasibility', pdf.page_no() + 4),
            ('5. Go-to-Market Strategy', pdf.page_no() + 5),
            ('6. Risk Analysis', pdf.page_no() + 6),
            ('7. Regulatory Compliance', pdf.page_no() + 7),
            ('8. Technology Infrastructure', pdf.page_no() + 8),
            ('9. Critical Review', pdf.page_no() + 9),
            ('10. Final Recommendations', pdf.page_no() + 10),
        ]
        
        for item, page in toc_items:
            pdf.set_font('Arial', '', 12)
            pdf.cell(0, 8, f'{item}', 0, 1)
        
        # Analysis Sections
        sections = [
            ('Strategic Planning', plan),
            ('Market Analysis', market),
            ('Competition Analysis', competition),
            ('Financial Feasibility', financial),
            ('Go-to-Market Strategy', gtm),
            ('Risk Analysis', risks),
            ('Regulatory Compliance', regulatory),
            ('Technology Infrastructure', technology),
            ('Critical Review', critic),
            ('Final Recommendations', final_report)
        ]
        
        for title, content in sections:
            if content and content.strip():
                pdf.add_page()
                pdf.chapter_title(title)
                
                # Handle very long content by breaking into subsections
                if len(content) > 3000:
                    # Try to extract subsections
                    subsections = extract_sections(content)
                    if subsections:
                        for subsection in subsections:
                            if subsection['title'] and subsection['title'] != 'Introduction':
                                pdf.section_title(subsection['title'])
                            pdf.add_text(clean_text(subsection['content']))
                    else:
                        # Break into paragraphs
                        paragraphs = content.split('\n\n')
                        for para in paragraphs:
                            if para.strip():
                                pdf.add_text(clean_text(para))
                else:
                    pdf.add_text(clean_text(content))
        
        # Appendix - Business Idea Details
        pdf.add_page()
        pdf.chapter_title('Appendix: Business Concept')
        pdf.add_text(clean_text(idea))
        
        return pdf.output(dest="S").encode("latin-1")
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        # Return a simple error PDF
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Business Analysis Report", ln=1)
            pdf.set_font("Arial", "", 12)
            pdf.ln(10)
            pdf.multi_cell(0, 8, f"Business Idea: {idea}\n\nAn error occurred while generating the detailed PDF report. Please use the web interface to view the full analysis.\n\nError: {str(e)}")
            return pdf.output(dest="S").encode("latin-1")
        except:
            return None
