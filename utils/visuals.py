import re
import math
import matplotlib.pyplot as plt

def make_market_chart(market_analysis_text: str, filename: str = "market_chart.png"):
    """
    Generate a bar chart for TAM/SAM/SOM from the market analysis text.
    Extracts numeric values for TAM, SAM, SOM (assuming values given in analysis).
    Saves chart as PNG to the given filename.
    Returns the filename if successful, else None.
    """
    # Default values in case parsing fails
    tam_val = sam_val = som_val = None
    # Try to find numeric values for TAM, SAM, SOM using regex
    patterns = {
        'TAM': r'TAM\s*[:\-]\s*\$?([\d\.,]+)\s*([Bb]illion|[Mm]illion|B|M)?',
        'SAM': r'SAM\s*[:\-]\s*\$?([\d\.,]+)\s*([Bb]illion|[Mm]illion|B|M)?',
        'SOM': r'SOM\s*[:\-]\s*\$?([\d\.,]+)\s*([Bb]illion|[Mm]illion|B|M)?'
    }
    values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, market_analysis_text)
        if match:
            num_str = match.group(1)
            unit = match.group(2)
            # Remove commas and convert to float
            try:
                num = float(num_str.replace(',', ''))
            except:
                num = None
            # Convert unit to millions
            if num is not None:
                if unit:
                    unit = unit.lower()
                    if unit.startswith('b'):  # billion
                        num = num * 1000  # convert to millions
                    # if million or M, it's already in millions
                    # if no unit or recognized, assume already in base unit (millions)
                values[key] = num
    if len(values) == 3:
        tam_val = values['TAM']
        sam_val = values['SAM']
        som_val = values['SOM']
    else:
        # If not all found, return None (chart won't be generated)
        return None

    # Determine scale for labeling
    unit_label = "Millions"
    # If TAM is huge, switch to billions for readability
    if tam_val and tam_val >= 1000:
        unit_label = "Billions"
        tam_plot = tam_val / 1000.0
        sam_plot = sam_val / 1000.0
        som_plot = som_val / 1000.0
    else:
        tam_plot = tam_val
        sam_plot = sam_val
        som_plot = som_val

    # Create bar chart
    categories = ['TAM', 'SAM', 'SOM']
    values_plot = [tam_plot, sam_plot, som_plot]
    fig, ax = plt.subplots(figsize=(6,4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    ax.bar(categories, values_plot, color=colors)
    ax.set_title("Market Size Estimates")
    ax.set_ylabel(f"Market Size ({unit_label})")
    # Annotate values on top of bars
    for i, val in enumerate(values_plot):
        ax.text(i, val + (0.05 * max(values_plot)), f"{val:.2f}", ha='center', fontweight='bold')
    fig.tight_layout()
    # Save to file
    try:
        fig.savefig(filename)
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error saving chart: {e}")
        return None

def draw_agent_graph(filename: str = "agent_graph.png"):
    """
    Generate a visualization of the agent workflow DAG (Planner -> Analysts -> Critic -> Synthesizer).
    Saves the diagram as a PNG.
    """
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('off')  # no axes
    # Define positions for nodes
    positions = {
        "Planner": (0, 1.0),
        "Market Analysis": (-0.6, 0.8),
        "Competition Analysis": (-0.2, 0.8),
        "Financial Analysis": (0.2, 0.8),
        "GTM Strategy": (0.6, 0.8),
        "Risks Analysis": (1.0, 0.8),
        "Critic": (0.2, 0.5),
        "Synthesizer": (0.2, 0.2)
    }
    # Node display (text with box)
    node_props = dict(boxstyle="round,pad=0.3", fc="#DCEEFF", ec="black", lw=1)
    # Draw nodes
    for node, (x,y) in positions.items():
        ax.text(x, y, node, ha='center', va='center', bbox=node_props, fontsize=9)
    # Helper to draw arrow
    def draw_arrow(start, end):
        sx, sy = positions[start]
        ex, ey = positions[end]
        # Draw arrow from center of start to center of end, with some offset to avoid covering text
        ax.annotate("", xy=(ex, ey+0.02), xytext=(sx, sy-0.02),
                    arrowprops=dict(arrowstyle="->", color="gray"))
    # Draw edges/arrows
    # Planner to all analysis nodes
    analysis_nodes = ["Market Analysis", "Competition Analysis", "Financial Analysis", "GTM Strategy", "Risks Analysis"]
    for node in analysis_nodes:
        draw_arrow("Planner", node)
        # Each analysis node to Critic
        draw_arrow(node, "Critic")
    # Critic to Synthesizer
    draw_arrow("Critic", "Synthesizer")
    # Save diagram
    try:
        fig.savefig(filename)
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error saving graph image: {e}")
        return None
