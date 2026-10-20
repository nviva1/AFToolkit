"""Analyse the p53 mutation panel: calibrate on experimental ddG, then rank.

    python reproductions/p53_e285k/analyse.py [results-dir] [figure-dir]
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else SCR
df = pd.read_csv(os.path.join(SCR, "p53_results.csv"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
C = {"benchmark": "#2a78d6", "E285_series": "#eb6834",
     "salt_bridge_network": "#1baf7a", "other_hotspots": "#eda100",
     "double_additive": "#6250d6"}

# ---------------------------------------------------------------- calibration
cal = df.dropna(subset=["experimental_ddg"]).copy()
rho, p_rho = stats.spearmanr(cal.pred_ddg_ensemble, cal.experimental_ddg)
r, p_r = stats.pearsonr(cal.pred_ddg_ensemble, cal.experimental_ddg)
rho_pl, p_pl = stats.spearmanr(cal.delta_plddt_site, cal.experimental_ddg)
rho_pae, _ = stats.spearmanr(cal.delta_pae_site, cal.experimental_ddg)

print("=" * 72)
print(f"CALIBRATION on {len(cal)} p53 mutations with experimental ddG")
print("=" * 72)
print(f"  predicted ddG     vs experimental:  Spearman {rho:+.3f} (p={p_rho:.2g})"
      f"   Pearson {r:+.3f} (p={p_r:.2g})")
print(f"  delta-pLDDT@site  vs experimental:  Spearman {rho_pl:+.3f} (p={p_pl:.2g})")
print(f"  delta-PAE@site    vs experimental:  Spearman {rho_pae:+.3f}")
print(f"  experimental ddG range: {cal.experimental_ddg.min():.2f} .. "
      f"{cal.experimental_ddg.max():.2f} (negative = destabilising)")
print(f"  predicted   ddG range: {cal.pred_ddg_ensemble.min():.2f} .. "
      f"{cal.pred_ddg_ensemble.max():.2f}")

# Where does each measure place a severely destabilising mutation?
severe = cal[cal.experimental_ddg < -3]
mild = cal[cal.experimental_ddg > -1]
print(f"\n  severely destabilising (exp ddG < -3, n={len(severe)}):"
      f"  mean pred ddG {severe.pred_ddg_ensemble.mean():+.3f},"
      f"  mean dpLDDT {severe.delta_plddt_site.mean():+.2f}")
print(f"  mild/neutral       (exp ddG > -1, n={len(mild)}):"
      f"  mean pred ddG {mild.pred_ddg_ensemble.mean():+.3f},"
      f"  mean dpLDDT {mild.delta_plddt_site.mean():+.2f}")
sep = stats.mannwhitneyu(severe.delta_plddt_site, mild.delta_plddt_site,
                         alternative="less")
print(f"  Mann-Whitney dpLDDT severe<mild: U={sep.statistic:.0f}, p={sep.pvalue:.3g}")

# -------------------------------------------------------------------- panel
panel = df[df.group != "benchmark"].copy()
print("\n" + "=" * 72)
print("PANEL (no experimental ddG available)")
print("=" * 72)
cols = ["mutation", "group", "pred_ddg_ensemble", "delta_plddt_site",
        "delta_pae_site", "delta_mean_plddt"]
print(panel[cols].sort_values("delta_plddt_site").to_string(index=False))

# Reference percentile of each panel mutation against the calibration set
print("\nseverity of each panel mutation, as a percentile of the 42 measured ones")
print("(0% = most destabilising of the measured set, by the same predictor)")
for _, row in panel.sort_values("delta_plddt_site").iterrows():
    pct_pl = stats.percentileofscore(cal.delta_plddt_site, row.delta_plddt_site)
    pct_dg = stats.percentileofscore(cal.pred_ddg_ensemble, row.pred_ddg_ensemble)
    print(f"  {row.mutation:<14} dpLDDT {row.delta_plddt_site:+6.2f} "
          f"(pct {pct_pl:5.1f})   ddG {row.pred_ddg_ensemble:+6.3f} (pct {pct_dg:5.1f})")

# --------------------------------------------------------------------- figure
# per-site values for the rescue panel
import json  # noqa: E402
RD = os.path.join(SCR, "reports")
def site285(name):
    """Δ pLDDT at residue 285 from a per-mutation report, if one was kept."""
    import glob
    hits = glob.glob(os.path.join(RD, f"*_{name}.json"))
    if not hits:
        return float("nan")
    change = json.load(open(hits[0]))["confidence"]["change"]
    return change["delta_per_site_plddt"].get("285", float("nan"))

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.40, wspace=0.30)

def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

# (a) calibration scatter - one series, so no legend; title names it
ax = fig.add_subplot(gs[0, 0])
tidy(ax)
ax.scatter(cal.experimental_ddg, cal.pred_ddg_ensemble, s=46, c=C["benchmark"],
           edgecolor="white", linewidth=0.8, zorder=3)
ax.set_xlabel("experimental ΔΔG (kcal/mol)", color=INK2, fontsize=10)
ax.set_ylabel("predicted ΔΔG (kcal/mol)", color=INK2, fontsize=10)
ax.set_title(f"a  Adapter ΔΔG vs experiment (n=42)\n"
             f"ρ = {rho:+.2f}, r = {r:+.2f}",
             color=INK, fontsize=11, loc="left", fontweight="bold")

# (b) the paper's own metric
ax = fig.add_subplot(gs[0, 1])
tidy(ax)
ax.scatter(cal.experimental_ddg, cal.delta_plddt_site, s=46, c=C["benchmark"],
           edgecolor="white", linewidth=0.8, zorder=3)
ax.axhline(0, color=INK2, lw=1, ls="--", alpha=0.6)
ax.set_xlabel("experimental ΔΔG (kcal/mol)", color=INK2, fontsize=10)
ax.set_ylabel("Δ pLDDT at mutated site", color=INK2, fontsize=10)
ax.set_title(f"b  Δ pLDDT vs experiment (n=42)\n"
             f"ρ = {rho_pl:+.2f}  — the metric used in the paper",
             color=INK, fontsize=11, loc="left", fontweight="bold")

# (d) compensatory rescue at site 285
ax = fig.add_subplot(gs[0, 2])
tidy(ax)
names = ["E285K", "E285K\n+R273H", "E285K\n+K132E"]
vals = [site285("E285K"), site285("E285K_R273H"), site285("E285K_K132E")]
cols = [C["E285_series"], C["other_hotspots"], C["salt_bridge_network"]]
bars = ax.bar(names, vals, color=cols, width=0.62, zorder=3)
ax.axhline(0, color=INK2, lw=1)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v - 0.25, f"{v:+.2f}", ha="center",
            va="top", fontsize=9, color=INK2)
ax.set_ylabel("Δ pLDDT at site 285", color=INK2, fontsize=10)
ax.set_title("d  Charge-swap rescue\neffect on site 285 only",
             color=INK, fontsize=11, loc="left", fontweight="bold")
ax.margins(y=0.22)

# (c) ranked panel - categorical by group, direct value labels
ax = fig.add_subplot(gs[1, :])
tidy(ax)
pr = panel.sort_values("delta_plddt_site")
ypos = np.arange(len(pr))
ax.barh(ypos, pr.delta_plddt_site, color=[C[g] for g in pr.group],
        height=0.74, zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(pr.mutation, fontsize=9, color=INK)
ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("Δ pLDDT at mutated site   (more negative = more disruptive)",
              color=INK2, fontsize=10)
q10 = np.percentile(cal.delta_plddt_site, 10)
ax.axvline(q10, color=INK2, lw=1, ls=":", alpha=0.8)
ax.text(q10, len(pr) - 0.2, "10th pct of\nmeasured set", fontsize=8,
        color=INK2, va="top", ha="center")
for y, v in zip(ypos, pr.delta_plddt_site):
    ax.text(v + (-0.25 if v < 0 else 0.25), y, f"{v:+.1f}", va="center",
            ha="right" if v < 0 else "left", fontsize=8, color=INK2)
handles = [plt.Rectangle((0, 0), 1, 1, color=C[g])
           for g in pr.group.unique()]
ax.legend(handles, [g.replace("_", " ") for g in pr.group.unique()],
          frameon=False, fontsize=9, loc="upper left",
          bbox_to_anchor=(0.01, 0.99), labelcolor=INK2, ncol=2)
ax.set_title("c  Predicted disruption across the panel  (p53 DBD, 2OCJ chain A, 4 recycles)",
             color=INK, fontsize=11, loc="left", fontweight="bold")
ax.margins(x=0.14)
ax.set_ylim(-0.8, len(pr) + 0.9)

fig.suptitle("p53 DNA-binding domain: predicted impact of E285K and related variants",
             fontsize=13, color=INK, fontweight="bold", x=0.5, y=0.985)
out = os.path.join(OUTDIR, "p53_mutation_panel.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
print(f"\nfigure -> {out}")
