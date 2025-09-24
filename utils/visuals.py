import re
import matplotlib.pyplot as plt

def _parse_val(s: str):
    s = s.strip()
    m = re.match(r"([\d\.,]+)\s*([kKmMbB])?", s)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    # Normalize to Millions
    if unit == "b":
        num *= 1000.0
    elif unit == "k":
        num /= 1000.0
    return num

def extract_tam_sam_som(text: str):
    # find a line that contains TAM, SAM, SOM
    lines = [ln.strip() for ln in text.splitlines() if "TAM" in ln and "SAM" in ln and "SOM" in ln]
    if not lines:
        return (None, None, None)
    line = lines[-1]
    m = re.search(r"TAM\s*:\s*([^,]+)", line); tam = _parse_val(m.group(1)) if m else None
    m = re.search(r"SAM\s*:\s*([^,]+)", line); sam = _parse_val(m.group(1)) if m else None
    m = re.search(r"SOM\s*:\s*([^,\)]+)", line); som = _parse_val(m.group(1)) if m else None
    return (tam, sam, som)

def make_market_chart(market_text: str, filename: str = "tam_sam_som_chart.png"):
    tam, sam, som = extract_tam_sam_som(market_text)
    if tam is None or sam is None or som is None:
        return None
    vals = [tam, sam, som]; labels = ["TAM", "SAM", "SOM"]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, vals)
    ax.set_title("Market Size (Millions)")
    ax.set_ylabel("Millions")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02 if v else 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename

def draw_agent_graph(filename: str = "agent_workflow.png"):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.axis("off")
    boxes = [
        ("Planner", 0.10, 0.85),
        ("Market", 0.05, 0.65),
        ("Competition", 0.28, 0.65),
        ("Financials", 0.51, 0.65),
        ("GTM", 0.74, 0.65),
        ("Risks", 0.87, 0.65),
        ("Critic", 0.51, 0.43),
        ("Synthesizer", 0.51, 0.20),
    ]
    for label, x, y in boxes:
        ax.text(x, y, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F1FF", ec="#1B3B6F"))
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    arr(0.10, 0.83, 0.05, 0.67); arr(0.10, 0.83, 0.28, 0.67)
    arr(0.10, 0.83, 0.51, 0.67); arr(0.10, 0.83, 0.74, 0.67)
    arr(0.10, 0.83, 0.87, 0.67)
    for x in [0.05, 0.28, 0.51, 0.74, 0.87]:
        arr(x, 0.63, 0.51, 0.45)
    arr(0.51, 0.41, 0.51, 0.22)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return filename
