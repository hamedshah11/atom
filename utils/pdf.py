from fpdf import FPDF
import re

class MarkdownPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def _clean_text(self, text: str) -> str:
        """Clean text for PDF rendering - handle unicode and special chars"""
        if not text:
            return ""
        
        # Replace common unicode characters that cause issues
        replacements = {
            '\u2018': "'",  # Left single quote
            '\u2019': "'",  # Right single quote
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...',# Ellipsis
            '\u00a0': ' ',  # Non-breaking space
            '\u2022': '-',  # Bullet point
            '\u20b9': 'Rs', # Rupee symbol
            '\u00a3': 'GBP',# Pound
            '\u20ac': 'EUR',# Euro
            '\u00b0': ' degrees', # Degree symbol
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove any remaining non-latin1 characters
        text = text.encode('latin-1', errors='ignore').decode('latin-1')
        
        return text
        
    def add_markdown_text(self, text: str):
        """Simple markdown to PDF converter with unicode handling"""
        if not text:
            return
            
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:
                self.ln(4)
                continue
            
            # Clean the line for PDF rendering
            line = self._clean_text(line)
                
            # Headers
            if line.startswith('###'):
                self.set_font("Arial", "B", 12)
                self.multi_cell(0, 7, line[3:].strip(), ln=1)
            elif line.startswith('##'):
                self.set_font("Arial", "B", 14)
                self.multi_cell(0, 8, line[2:].strip(), ln=1)
            elif line.startswith('#'):
                self.set_font("Arial", "B", 16)
                self.multi_cell(0, 9, line[1:].strip(), ln=1)
            # Bullet points
            elif line.startswith('-') or line.startswith('*'):
                self.set_font("Arial", "", 11)
                # Indent bullet points
                self.cell(5, 6, "", ln=0)  # Indent
                self.multi_cell(0, 6, line, ln=1)
            # Bold text (simple version)
            elif '**' in line:
                self.set_font("Arial", "B", 11)
                # Simple bold replacement
                clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                self.multi_cell(0, 6, clean_line, ln=1)
            # Regular text
            else:
                self.set_font("Arial", "", 11)
                # Handle long lines that might cause issues
                try:
                    self.multi_cell(0, 6, line, ln=1)
                except Exception as e:
                    # If multi_cell fails, just skip this line and log it
                    print(f"Warning: Could not add line to PDF: {str(e)[:100]}")
                    continue

def generate_analysis_pdf(
    idea: str, plan: str, market: str, competition: str, financial: str,
    gtm: str, risks: str, critic: str, final_report: str
) -> bytes | None:
    """
    Generate PDF with robust error handling and unicode support
    """
    try:
        pdf = MarkdownPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 12, "Business Idea Analysis Report", ln=1, align='C')
        pdf.ln(5)
        
        # Business Idea
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Business Idea", ln=1)
        pdf.set_font("Arial", "", 11)
        cleaned_idea = pdf._clean_text(idea)
        pdf.multi_cell(0, 6, cleaned_idea)
        pdf.ln(4)
        
        # Analysis sections
        sections = [
            ("Analysis Plan", plan),
            ("Market Analysis", market),
            ("Competition Analysis", competition),
            ("Financial Feasibility", financial),
            ("Go-to-Market Strategy", gtm),
            ("Risk Analysis", risks),
            ("Critical Review", critic),
            ("Executive Summary & Synthesis", final_report)
        ]
        
        for title, content in sections:
            if content and content.strip():
                try:
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, title, ln=1)
                    pdf.ln(3)
                    
                    # Use markdown parser for final report, regular text for others
                    if title == "Executive Summary & Synthesis":
                        pdf.add_markdown_text(content)
                    else:
                        pdf.set_font("Arial", "", 11)
                        cleaned_content = pdf._clean_text(content)
                        # Split into smaller chunks to avoid issues
                        for paragraph in cleaned_content.split('\n'):
                            if paragraph.strip():
                                pdf.multi_cell(0, 6, paragraph.strip(), ln=1)
                                pdf.ln(2)
                    pdf.ln(4)
                except Exception as section_error:
                    print(f"Error adding section '{title}': {section_error}")
                    # Add error note but continue with other sections
                    pdf.set_font("Arial", "I", 10)
                    pdf.multi_cell(0, 6, f"[Content could not be rendered]", ln=1)
                    continue
        
        # Footer on last page
        pdf.ln(10)
        pdf.set_font("Arial", "I", 10)
        pdf.cell(0, 6, "Generated by Business Idea Analyzer - Powered by GPT-4", ln=1, align='C')
        
        # Output as bytes
        pdf_output = pdf.output(dest="S")
        
        # Convert to bytes properly
        if isinstance(pdf_output, bytes):
            return pdf_output
        elif isinstance(pdf_output, str):
            return pdf_output.encode("latin-1", errors='ignore')
        else:
            return bytes(pdf_output)
        
    except Exception as e:
        print(f"Critical PDF generation error: {type(e).__name__}: {str(e)}")
        
        # Return a simple fallback PDF with error details
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Business Idea Analysis Report", ln=1)
            pdf.ln(5)
            
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Business Idea:", ln=1)
            pdf.set_font("Arial", "", 11)
            
            # Safely add the idea
            safe_idea = idea.encode('latin-1', errors='ignore').decode('latin-1')
            pdf.multi_cell(0, 6, safe_idea[:500])
            pdf.ln(5)
            
            # Error message
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 6, 
                f"PDF generation encountered an error: {str(e)[:200]}\n\n"
                "Please copy the analysis from the web interface.\n\n"
                "Common causes:\n"
                "- Special unicode characters in content\n"
                "- Very long text sections\n"
                "- Unsupported formatting"
            )
            
            pdf_output = pdf.output(dest="S")
            if isinstance(pdf_output, bytes):
                return pdf_output
            else:
                return pdf_output.encode("latin-1", errors='ignore')
        except:
            # Last resort - return None
            print("Even fallback PDF generation failed")
            return None
