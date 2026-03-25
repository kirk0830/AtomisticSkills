"""
Plot BDE comparison for methanol (CO) using MACE-OMOL-extra-large.
Shows both homolytic and heterolytic BDEs per bond.
Also investigates whether MACE-OMOL differentiates charge states in gas phase.

# Env: mace-agent
"""
import json
import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.insert(0, project_root)

# Load plot standards style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

EXAMPLE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(EXAMPLE_DIR, "bde_results.json")
OUT_FILE = os.path.join(EXAMPLE_DIR, "bde_comparison.png")

with open(DATA_FILE) as f:
    data = json.load(f)

# Build per-bond records
bonds = []
for b in data["bonds"]:
    if b.get("skipped"):
        continue
    syms = b["atom_symbols"]
    idxs = b["atom_indices"]
    label = f"{syms[0]}-{syms[1]}\n({idxs[0]}-{idxs[1]})"
    
    homo = b.get("bde_kcal_mol")
    hetero = b.get("heterolytic_bde_kcal_mol")

    # Heterolytic variants (both polarity directions)
    variants = b.get("heterolytic_variants", [])
    v1 = variants[0]["bde_kcal_mol"] if len(variants) > 0 else None
    v2 = variants[1]["bde_kcal_mol"] if len(variants) > 1 else None

    bonds.append({
        "label": label,
        "homo": homo,
        "hetero": hetero,
        "variant_A": v1,
        "variant_B": v2,
        "frag1_formula": b.get("frag1_formula", ""),
        "frag2_formula": b.get("frag2_formula", ""),
    })

labels = [b["label"] for b in bonds]
x = np.arange(len(labels))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ── Left panel: Homo vs Hetero (best) ──────────────────────────────────────
ax = axes[0]
homo_vals = [b["homo"] for b in bonds]
hetero_vals = [b["hetero"] for b in bonds]

bars1 = ax.bar(x - width/2, homo_vals, width, label="Homolytic (radical)", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x + width/2, hetero_vals, width, label="Heterolytic (best ionic)", color="#FF9800", alpha=0.85)

# label values
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.0f}", ha="center", va="bottom", fontsize=8, color="#E65100")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("BDE (kcal/mol)", fontweight="bold")
ax.set_title("Methanol: Homolytic vs Heterolytic BDE\n(MACE-OMOL-extra-large, gas phase)", fontsize=11, fontweight="bold")
ax.legend(framealpha=0.7)
ax.set_ylim(0, max(homo_vals + hetero_vals) * 1.15)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Annotate delta
for xi, bh, bhe in zip(x, homo_vals, hetero_vals):
    diff = abs(bh - bhe)
    ax.text(xi, max(bh, bhe) + 14, f"Δ={diff:.1f}", ha="center", fontsize=7.5, color="gray")

# ── Right panel: Heterolytic polarity variants ─────────────────────────────
ax2 = axes[1]
v1_vals = [b["variant_A"] for b in bonds]  # frag1+ / frag2-
v2_vals = [b["variant_B"] for b in bonds]  # frag1- / frag2+
frag_labels = [f"{b['frag1_formula']}–{b['frag2_formula']}" for b in bonds]

bars3 = ax2.bar(x - width/2, v1_vals, width, label="frag1⁺ + frag2⁻", color="#E91E63", alpha=0.85)
bars4 = ax2.bar(x + width/2, v2_vals, width, label="frag1⁻ + frag2⁺", color="#9C27B0", alpha=0.85)

for bar, val in zip(bars3, v1_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.0f}",
             ha="center", va="bottom", fontsize=8)
for bar, val in zip(bars4, v2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{val:.0f}",
             ha="center", va="bottom", fontsize=8)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylabel("Heterolytic BDE (kcal/mol)", fontweight="bold")
ax2.set_title("Methanol: Heterolytic Polarity Variants\n(MACE-OMOL-extra-large, gas phase)", fontsize=11, fontweight="bold")
ax2.legend(framealpha=0.7)
ax2.set_ylim(0, max(v1_vals + v2_vals) * 1.15)
ax2.yaxis.grid(True, alpha=0.3)
ax2.set_axisbelow(True)

# Note about charge invariance
fig.text(
    0.5, 0.01,
    "Note: MACE-OMOL produces charge-invariant fragment energies in gas phase — "
    "homo and hetero BDEs are identical because isolated fragment energies\n"
    "do not change with the charge/spin annotation (see README for discussion).",
    ha="center", fontsize=8.5, color="#666666", style="italic"
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_FILE}")
