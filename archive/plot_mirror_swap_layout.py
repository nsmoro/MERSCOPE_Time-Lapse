#!/usr/bin/env python
"""
Whole-slide view with the live data mirrored and its fields swapped back.

Applies, per region: a vertical mirror of the block, then the fields put
back into their original slots. Net effect is that each field keeps its
position while its contents flip top-to-bottom -- what you would see if
each field's y axis ran opposite to the stage's.

Draws the result beside MERSCOPE, coloured by region, and reports the
median nearest-neighbour distance before and after so the two can be
compared by number as well as by eye.

    python plot_mirror_swap_layout.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --flip xy --mirror y --reorder y --out ./mirror_swap_layout
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy")
ap.add_argument("--mirror", default="y", choices=["none", "x", "y", "xy"])
ap.add_argument("--reorder", default="y", choices=["none", "x", "y"])
ap.add_argument("--point-size", type=float, default=1.6)
ap.add_argument("--out", default="./mirror_swap_layout")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
sx = -1 if "x" in a.flip else 1
sy = -1 if "y" in a.flip else 1

regions = [r.strip() for r in a.region_order.split(",")]
positions = sorted(live.Position.unique())
groups = {}
for k, reg in enumerate(regions):
    lo = k * a.per_region
    hi = (k + 1) * a.per_region if k < len(regions) - 1 else len(positions)
    groups[reg] = positions[lo:hi]

live = live.assign(px=live[lx] * sx, py=live[ly] * sy)
live["region"] = None
for reg, ps in groups.items():
    live.loc[live.Position.isin(ps), "region"] = reg
live = live[live.region.notna()].copy()

def transform_block(df):
    out = df.copy()
    if "x" in a.mirror:
        c = (out.px.min() + out.px.max()) / 2
        out["px"] = 2 * c - out.px
    if "y" in a.mirror:
        c = (out.py.min() + out.py.max()) / 2
        out["py"] = 2 * c - out.py
    if a.reorder != "none":
        ai = "px" if a.reorder == "x" else "py"
        cen = out.groupby("Position")[["px", "py"]].mean()
        order = cen[ai].sort_values().index.tolist()
        for src, dst in zip(order, order[::-1]):
            sh = cen.loc[dst] - cen.loc[src]
            m = out.Position == src
            out.loc[m, "px"] += sh["px"]
            out.loc[m, "py"] += sh["py"]
    return out

before, after, stats = [], [], []
for reg, g in live.groupby("region"):
    t = transform_block(g)
    before.append(g); after.append(t)
    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    if len(B) < 10:
        continue
    Bc = B - (B.min(0) + B.max(0)) / 2
    tree = cKDTree(Bc)
    out = {}
    for name, dd in (("before", g), ("after", t)):
        P = dd[["px", "py"]].to_numpy(float)
        P = P - (P.min(0) + P.max(0)) / 2
        d, _ = tree.query(P, k=1)
        out[name] = (float(np.median(d)), float((d < 10).mean()))
    stats.append(dict(region=reg, n=len(g),
                      median_before=out["before"][0], median_after=out["after"][0],
                      pct10_before=out["before"][1], pct10_after=out["after"][1]))

before = pd.concat(before); after = pd.concat(after)
S = pd.DataFrame(stats)
print(f"{'region':>7} {'median before':>15} {'median after':>14} {'change':>9}")
print("-" * 50)
for _, r in S.iterrows():
    ch = r.median_after - r.median_before
    print(f"{r.region:>7} {r.median_before:>14.1f} {r.median_after:>14.1f} "
          f"{ch:>+9.1f}")
print(f"\n  overall median: before {S.median_before.median():.1f} um, "
      f"after {S.median_after.median():.1f} um")
print(f"  improved in {(S.median_after < S.median_before).sum()}/{len(S)} regions")
S.to_csv(f"{a.out}/mirror_swap_stats.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cmap = plt.get_cmap("tab10")
colour = {reg: cmap(k % 10) for k, reg in enumerate(groups)}

fig, ax = plt.subplots(1, 2, figsize=(22, 10))
for reg in groups:
    s = after[after.region == reg]
    if len(s):
        ax[0].scatter(s.px, s.py, s=a.point_size, color=colour[reg],
                      linewidths=0)
        ax[0].annotate(f"{reg}\n({groups[reg][0]}-{groups[reg][-1]})",
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
ax[0].set_title(f"TIME-LAPSE  ({len(after):,} cells, '{a.flip}', "
                f"mirrored {a.mirror}, fields swapped on {a.reorder})",
                fontsize=13)
ax[1].set_title(f"MERSCOPE  ({len(mer):,} cells, as acquired)", fontsize=13)
for A_ in ax:
    A_.set_aspect("equal"); A_.invert_yaxis()
    A_.set_xlabel("um"); A_.grid(alpha=.15, lw=.5)
ax[0].set_ylabel("um")
fig.suptitle("Matching colours are the same wells", fontsize=15, y=0.97)
fig.tight_layout(rect=[0, 0, 1, 0.94])
f = f"{a.out}/layout_mirror_swap.png"
fig.savefig(f, dpi=150); plt.close(fig)
print(f"\n  -> {f}")
print(f"  -> {a.out}/mirror_swap_stats.csv")
