#!/usr/bin/env python
"""
One figure per region: a locator strip plus every tile, on a single page.

The top row shows the whole slide with the region outlined, and the
region itself with each tile boxed and numbered, so it is always clear
which part of the slide a panel is showing. Below that, the tiles
themselves: live on the left of each pair, MERSCOPE on the right, with
matched cells carrying the same number in both.

Nine figures for nine regions, and the tile counts add up to the whole
match table.

    python region_overview_figure.py \
        --pairs ./pairs_swapped/pairs_gate52um.csv \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --tiles 3 --out ./region_figures
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--tiles", type=int, default=3)
ap.add_argument("--label-limit", type=int, default=90)
ap.add_argument("--out", default="./region_figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

P = pd.read_csv(a.pairs, comment="#")
mer = pd.read_csv(a.merscope)
B_all = mer[["center_x", "center_y"]].to_numpy(float)
A_all = np.c_[P.live_x, P.live_y]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

TEAL, PINK, GREY = "#1D9E75", "#B03A8B", "#D3D9DD"
index = []

for reg, g in P.groupby("region"):
    g = g.reset_index(drop=True)
    g["label"] = np.arange(1, len(g) + 1)
    nt = a.tiles
    g["tx"] = pd.qcut(g.mer_x, nt, labels=False, duplicates="drop")
    g["ty"] = pd.qcut(g.mer_y, nt, labels=False, duplicates="drop")
    g["tile"] = g.ty * nt + g.tx + 1
    tiles = sorted(t for t, s in g.groupby("tile") if len(s) >= 2)
    if not tiles:
        continue

    ncol = min(3, len(tiles))
    nrow = int(np.ceil(len(tiles) / ncol))
    fig = plt.figure(figsize=(6.6 * ncol, 4.2 + 4.6 * nrow))
    gs = GridSpec(nrow + 1, ncol * 2, figure=fig,
                  height_ratios=[0.85] + [1] * nrow, hspace=0.33, wspace=0.18)

    # ---- locator: whole slide, region outlined ----
    ax = fig.add_subplot(gs[0, :ncol])
    ax.scatter(B_all[:, 0], B_all[:, 1], s=.6, c=GREY, linewidths=0)
    gm = mer[mer.region == reg]
    ax.scatter(gm.center_x, gm.center_y, s=.8, c=PINK, linewidths=0)
    x0, x1 = gm.center_x.min(), gm.center_x.max()
    y0, y1 = gm.center_y.min(), gm.center_y.max()
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec="#1D6F8C", lw=2.4))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"whole slide — {reg} outlined", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- the region, with tiles boxed ----
    ax = fig.add_subplot(gs[0, ncol:])
    ax.scatter(gm.center_x, gm.center_y, s=2.2, c=GREY, linewidths=0)
    ax.scatter(g.mer_x, g.mer_y, s=5, c=PINK, linewidths=0)
    for t in tiles:
        s = g[g.tile == t]
        bx0, bx1 = s.mer_x.min(), s.mer_x.max()
        by0, by1 = s.mer_y.min(), s.mer_y.max()
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False,
                               ec="#1D6F8C", lw=1.6))
        ax.annotate(str(int(t)), ((bx0 + bx1) / 2, (by0 + by1) / 2),
                    fontsize=14, weight="bold", color="#1D6F8C",
                    ha="center", va="center",
                    bbox=dict(boxstyle="circle,pad=0.22", fc="white",
                              ec="#1D6F8C", lw=1.4, alpha=.9))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{reg} — {len(g)} matched pairs in {len(tiles)} tiles",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    # ---- the tiles ----
    for k, t in enumerate(tiles):
        s = g[g.tile == t]
        show = len(s) <= a.label_limit
        r, c = divmod(k, ncol)
        for side, (xc, yc, colour, bg, nm) in enumerate(
                [("live_x", "live_y", TEAL, A_all, "live"),
                 ("mer_x", "mer_y", PINK, B_all, "MERSCOPE")]):
            ax = fig.add_subplot(gs[r + 1, c * 2 + side])
            px0, px1 = s[xc].min(), s[xc].max()
            py0, py1 = s[yc].min(), s[yc].max()
            pad = max(px1 - px0, py1 - py0) * 0.1 + 25
            m = ((bg[:, 0] > px0 - pad) & (bg[:, 0] < px1 + pad) &
                 (bg[:, 1] > py0 - pad) & (bg[:, 1] < py1 + pad))
            ax.scatter(bg[m, 0], bg[m, 1], s=11, c=GREY, linewidths=0)
            ax.scatter(s[xc], s[yc], s=30, c=colour, zorder=3)
            if show:
                for _, rr in s.iterrows():
                    ax.annotate(int(rr.label), (rr[xc], rr[yc]), fontsize=6,
                                weight="bold", color="#111", zorder=4,
                                xytext=(4, 3), textcoords="offset points")
            ax.set_aspect("equal"); ax.invert_yaxis()
            ax.set_xlim(px0 - pad, px1 + pad); ax.set_ylim(py1 + pad, py0 - pad)
            ax.set_xticks([]); ax.set_yticks([])
            ttl = f"tile {int(t)} · {nm}"
            if side == 0:
                ttl += f"  ({len(s)} pairs{'' if show else ', unlabelled'})"
            ax.set_title(ttl, fontsize=9.5)

    med = g.dist_um.median() if "dist_um" in g else np.nan
    fig.suptitle(f"{reg} — {len(g)} matched pairs, median separation "
                 f"{med:.0f} um.  Same number is the same pair; "
                 f"grey cells are unmatched.", fontsize=14, y=0.995)
    f = f"{a.out}/{reg}.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    g.to_csv(f"{a.out}/{reg}.csv", index=False)
    index.append(dict(region=reg, pairs=len(g), tiles=len(tiles),
                      median_um=med, file=f))
    print(f"  {reg}: {len(g):5d} pairs in {len(tiles)} tiles  -> {f}")

S = pd.DataFrame(index); S.to_csv(f"{a.out}/index.csv", index=False)
print(f"\n  {S.pairs.sum():,} pairs across {len(S)} figures, one per region")
print(f"  table has {len(P):,} pairs")
print(f"\n  -> {a.out}/index.csv")
