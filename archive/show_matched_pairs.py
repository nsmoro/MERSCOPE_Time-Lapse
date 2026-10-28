#!/usr/bin/env python
"""
Zoomed side-by-side with each matched pair given the same number in both
panels — the visual check on what the match table actually contains.

Live on the left, MERSCOPE on the right, at matched scale. Cell 7 on the
left is paired with cell 7 on the right. If the matching is real, the two
numbered cells sit in the same place within their respective patches and
their neighbours correspond too. If it is not, the numbers will be
scattered differently in the two panels.

Also reports what the match rate means from the time-lapse side: of the
live cells, how many have a MERSCOPE partner within the radius, and how
that compares with scattering the same cells at random over the same area.

    python show_matched_pairs.py \
        --pairs ./pairs_swapped/pairs_gate65um.csv \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region R18 --zoom-um 600 --out ./matched_view
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region", default=None, help="restrict to one region")
ap.add_argument("--zoom-um", type=float, default=600.0)
ap.add_argument("--n-views", type=int, default=3)
ap.add_argument("--max-labels", type=int, default=60)
ap.add_argument("--out", default="./matched_view")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

P = pd.read_csv(a.pairs, comment="#")
live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
if a.region:
    P = P[P.region == a.region].reset_index(drop=True)
if not len(P):
    raise SystemExit("no pairs after filtering")

need = {"live_x", "live_y", "mer_x", "mer_y"}
if not need <= set(P.columns):
    raise SystemExit(f"pairs file needs {sorted(need)}")

# ----------------------------------------------------- what the rate means
gate = float(np.ceil(P.dist_um.max())) if "dist_um" in P else a.zoom_um
n_live = len(live) if not a.region else None
print("=" * 64)
print("WHAT THE MATCH RATE MEANS, FROM THE TIME-LAPSE SIDE")
print("=" * 64)
tot_live = len(live)
print(f"  live cells at the last frame        {tot_live:,}")
print(f"  of those, paired to a MERSCOPE cell {len(P):,}  "
      f"({len(P)/tot_live:.1%})")
print(f"  search radius used                  {gate:.0f} um")

B_all = mer[["center_x", "center_y"]].to_numpy(float)
tree_all = cKDTree(B_all)
rng = np.random.default_rng(0)
lo, hi = B_all.min(0), B_all.max(0)
ch = np.mean([(tree_all.query(rng.uniform(lo, hi, (tot_live, 2)), k=1)[0]
               < gate).mean() for _ in range(5)])
print(f"\n  the same cells scattered at random  {ch:.1%}")
print(f"  enrichment over chance              "
      f"{(len(P)/tot_live)/max(ch,1e-12):.2f}x")
print(f"\n  So of the {len(P)/tot_live:.1%}, roughly {ch:.1%} would appear")
print(f"  even with no correspondence at all, because at {gate:.0f} um a")
print(f"  live cell has several MERSCOPE cells within reach.")

# ----------------------------------------------------- labelled views
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = P.sort_values("dist_um").reset_index(drop=True) if "dist_um" in P else P
seeds, used = [], []
for _, r in P.iterrows():
    if all(np.hypot(r.live_x - u[0], r.live_y - u[1]) > a.zoom_um * 1.5
           for u in used):
        seeds.append(r); used.append((r.live_x, r.live_y))
    if len(seeds) >= a.n_views:
        break

for k, s in enumerate(seeds, 1):
    h = a.zoom_um / 2
    sel = P[(np.abs(P.live_x - s.live_x) < h) &
            (np.abs(P.live_y - s.live_y) < h)].copy()
    if len(sel) > a.max_labels:
        sel = sel.head(a.max_labels)
    if len(sel) < 3:
        continue
    sel = sel.reset_index(drop=True)
    sel["label"] = np.arange(1, len(sel) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(19, 9.5))
    for side, (xc, yc, name, colour) in enumerate(
            [("live_x", "live_y", "TIME-LAPSE", "#1D9E75"),
             ("mer_x", "mer_y", "MERSCOPE", "#B03A8B")]):
        cx, cy = sel[xc].mean(), sel[yc].mean()
        if side == 0:
            bg = np.c_[P.live_x, P.live_y]
        else:
            bg = B_all
        m = (np.abs(bg[:, 0] - cx) < h * 1.3) & (np.abs(bg[:, 1] - cy) < h * 1.3)
        ax[side].scatter(bg[m, 0], bg[m, 1], s=26, c="#CFD6DB", linewidths=0,
                         zorder=1)
        ax[side].scatter(sel[xc], sel[yc], s=70, c=colour, zorder=3)
        for _, r in sel.iterrows():
            ax[side].annotate(int(r.label), (r[xc], r[yc]), fontsize=9,
                              weight="bold", color="#111", zorder=4,
                              xytext=(7, 6), textcoords="offset points")
        ax[side].set_aspect("equal"); ax[side].invert_yaxis()
        ax[side].set_xlim(cx - h * 1.3, cx + h * 1.3)
        ax[side].set_ylim(cy + h * 1.3, cy - h * 1.3)
        ax[side].set_title(f"{name} — {len(sel)} matched cells numbered",
                           fontsize=13)
        ax[side].set_xlabel("um"); ax[side].grid(alpha=.2, lw=.5)
    ax[0].set_ylabel("um")
    med = sel.dist_um.median() if "dist_um" in sel else np.nan
    fig.suptitle(f"View {k} — same number means the same pair "
                 f"(median separation {med:.0f} um). Grey cells are "
                 f"unmatched neighbours.", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = f"{a.out}/matched_view_{k}.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"\n  -> {f}")
    sel.to_csv(f"{a.out}/matched_view_{k}.csv", index=False)

print("\n  Compare the two panels: for a real match, cell 7 sits in the")
print("  same place within its patch on both sides, and the cells around")
print("  it correspond too.")
