#!/usr/bin/env python
"""
Time-lapse and MERSCOPE side by side, in the orientation from the bench.

Applies the stated orientation to the live data and colours both datasets
by the region each belongs to, so matching colours should be the same
physical wells. This is the picture to check against the instrument's own
slide overview.

    python plot_marcus_layout.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region-order R9,R11,R12,R13,R14,R16,R17,R18,R20 \
        --flip xy --out ./marcus_layout
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region-order", required=True)
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy", help="none|x|y|xy applied to live")
ap.add_argument("--groups", default=None,
                help="explicit, e.g. R9:1-3,R20:25-36")
ap.add_argument("--point-size", type=float, default=1.6)
ap.add_argument("--out", default="./marcus_layout")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
regions = [r.strip() for r in a.region_order.split(",")]
positions = sorted(live.Position.unique())

if a.groups:
    groups = {}
    for part in a.groups.split(","):
        reg, rng = part.split(":")
        lo, hi = (int(v) for v in rng.split("-"))
        groups[reg.strip()] = list(range(lo, hi + 1))
else:
    groups = {}
    for k, reg in enumerate(regions):
        lo = k * a.per_region
        hi = (k + 1) * a.per_region if k < len(regions) - 1 else len(positions)
        groups[reg] = positions[lo:hi]

sx = -1 if "x" in a.flip else 1
sy = -1 if "y" in a.flip else 1
live = live.assign(px=live[lx]*sx, py=live[ly]*sy)
print(f"live oriented with flip '{a.flip}'")
for reg, ps in groups.items():
    n = live.Position.isin(ps).sum()
    m = (mer.region == reg).sum()
    print(f"  {reg:>5}  positions {ps[0]:>2}-{ps[-1]:<2}  "
          f"{n:>5} live cells   {m:>5} MERSCOPE cells")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")
colour = {reg: cmap(k % 10) for k, reg in enumerate(groups)}

fig, ax = plt.subplots(1, 2, figsize=(22, 10))
for reg, ps in groups.items():
    s = live[live.Position.isin(ps)]
    if len(s):
        ax[0].scatter(s.px, s.py, s=a.point_size, color=colour[reg],
                      linewidths=0)
        ax[0].annotate(f"{reg}\n({ps[0]}-{ps[-1]})",
                       (s.px.mean(), s.py.mean()), fontsize=11, weight="bold",
                       ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                 ec=colour[reg], lw=1.6, alpha=.9))
    g = mer[mer.region == reg]
    if len(g):
        ax[1].scatter(g.center_x, g.center_y, s=a.point_size,
                      color=colour[reg], linewidths=0)
        ax[1].annotate(reg, (g.center_x.mean(), g.center_y.mean()),
                       fontsize=12, weight="bold", ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                 ec=colour[reg], lw=1.6, alpha=.9))

ax[0].set_title(f"TIME-LAPSE  ({len(live):,} cells, oriented '{a.flip}')",
                fontsize=14)
ax[1].set_title(f"MERSCOPE  ({len(mer):,} cells, as acquired)", fontsize=14)
for A in ax:
    A.set_aspect("equal"); A.invert_yaxis()
    A.set_xlabel("um"); A.grid(alpha=.15, lw=.5)
ax[0].set_ylabel("um")
fig.suptitle("Matching colours are the same wells, per the bench record",
             fontsize=15, y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.94])
f = f"{a.out}/marcus_layout.png"
fig.savefig(f, dpi=150); print(f"\n  -> {f}")
