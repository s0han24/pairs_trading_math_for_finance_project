def parse_prettytable(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Keep only rows with '|'
    rows = [line for line in lines if line.startswith("|")]
    
    # Split cells
    table = []
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        table.append(cells)
    
    return table


def to_markdown(table):
    header = table[0]
    rows = table[1:]
    
    md = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    for row in rows:
        md.append("| " + " | ".join(row) + " |")
    
    return "\n".join(md)


def to_latex(table):
    header = table[0]
    rows = table[1:]
    
    cols = " | ".join(["c"] * len(header))
    
    latex = []
    latex.append("\\begin{tabular}{" + cols + "}")
    latex.append("\\hline")
    latex.append(" & ".join(header) + " \\\\")
    latex.append("\\hline")
    
    for row in rows:
        latex.append(" & ".join(row) + " \\\\")
    
    latex.append("\\hline")
    latex.append("\\end{tabular}")
    
    return "\n".join(latex)


def convert_file(filepath, mode="markdown"):
    with open(filepath, "r") as f:
        text = f.read()
    
    table = parse_prettytable(text)
    
    if mode == "markdown":
        output = to_markdown(table)
    elif mode == "latex":
        output = to_latex(table)
    else:
        raise ValueError("mode must be 'markdown' or 'latex'")
    
    with open(filepath, "w") as f:
        f.write(output)


# Usage:
convert_file("table.txt", mode="markdown")  # or "latex"
