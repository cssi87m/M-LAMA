"""
Script to update temp.ipynb: adds PowerNorm(gamma=0.5) to the confusion matrix heatmap
for better color distribution in the Greens colormap.
"""
import json

notebook_path = "/home/user06/Interspeech_2026/Model/Model/Preds_fluency/temp.ipynb"

with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    # Find the cell with the heatmap code (contains sns.heatmap and LinearSegmentedColormap)
    source_str = "".join(source)
    if "LinearSegmentedColormap" in source_str and "sns.heatmap" in source_str:
        new_source = []
        for line in source:
            # Fix import: add PowerNorm
            if "from matplotlib.colors import LinearSegmentedColormap" in line and "PowerNorm" not in line:
                line = line.replace(
                    "from matplotlib.colors import LinearSegmentedColormap",
                    "from matplotlib.colors import LinearSegmentedColormap, PowerNorm"
                )
            # Add norm parameter after vmin=0
            if "vmin=0," in line and "norm=" not in source_str:
                line = line.rstrip("\n") + "\n"
                new_source.append(line)
                new_source.append("                 norm=PowerNorm(gamma=0.5),\n")
                continue
            new_source.append(line)
        cell["source"] = new_source
        # Clear outputs so the notebook looks clean
        cell["outputs"] = []
        cell["execution_count"] = None
        print("✓ Updated heatmap cell with PowerNorm(gamma=0.5)")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print(f"✓ Saved updated notebook to {notebook_path}")
