#!/usr/bin/env python
"""
One labelled figure per region, covering every matched pair in it.

Unlike a few sample windows, this partitions the matches: each region
gets its own page, and every pair in that region is drawn and numbered.
Add up the per-figure counts and you get the whole table back.

Labels are only printed where they would be readable. Where a region
holds more pairs than can be numbered legibly, the figure says how many
are shown of how many, and a tiled version splits the region into panels
so that all of them are labelled somewhere.

    python matched_by_region.py \
        --pairs ./pairs_swapped/pairs_gate52um.csv \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --out ./by_region

    # split each region into tiles so every pair gets a readable label
    python matched_by_region.py ... --tiles 3
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--tiles", type=int, default=1,
                help="split each region into N x N tiles; 1 = whole region "
                     "on one page")
ap.add_argument("--label-limit", type=int, default=120,
                help="above this many pairs in a panel, labels are omitted "
                     "as unreadable")
ap.add_argument("--out", default="./by_region")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

P = pd.read_csv(a.pairs, comment="#")
live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
if "region" not in P:
    raise SystemExit("pairs file has no region column")

B_all = mer[["center_x", "center_y"]].to_numpy(float)
A_all = np.c_[P.live_x, P.live_y]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

total_shown = 0
rows = []
for reg, g in P.groupby("region"):
    g = g.reset_index(drop=True)
    g["label"] = np.arange(1, len(g) + 1)

    # tile the region if asked
    if a.tiles > 1:
        qx = pd.qcut(g.live_x, a.tiles, labels=False, duplicates="drop")
        qy = pd.qcut(g.live_y, a.tiles, labels=False, duplicates="drop")
        g["tile"] = qy * a.tiles + qx
    else:
        g["tile"] = 0

    for tile, t in g.groupby("tile"):
        if len(t) < 2:
            continue
        show_labels = len(t) <= a.label_limit
        fig, ax = plt.subplots(1, 2, figsize=(19, 9.5))
        for side, (xc, yc, name, colour, bg) in enumerate(
                [("live_x", "live_y", "TIME-LAPSE", "#1D9E75", A_all),
                 ("mer_x", "mer_y", "MERSCOPE", "#B03A8B", B_all)]):
            x0, x1 = t[xc].min(), t[xc].max()
            y0, y1 = t[yc].min(), t[yc].max()
            pad = max(x1 - x0, y1 - y0) * 0.08 + 30
            m = ((bg[:, 0] > x0 - pad) & (bg[:, 0] < x1 + pad) &
                 (bg[:, 1] > y0 - pad) & (bg[:, 1] < y1 + pad))
            ax[side].scatter(bg[m, 0], bg[m, 1], s=14, c="#D3D9DD",
                             linewidths=0, zorder=1)
            ax[side].scatter(t[xc], t[yc], s=38, c=colour, zorder=3)
            if show_labels:
                for _, r in t.iterrows():
                    ax[side].annotate(int(r.label), (r[xc], r[yc]),
                                      fontsize=7, weight="bold", color="#111",
                                      zorder=4, xytext=(5, 4),
                                      textcoords="offset points")
            ax[side].set_aspect("equal"); ax[side].invert_yaxis()
            ax[side].set_xlim(x0 - pad, x1 + pad)
            ax[side].set_ylim(y1 + pad, y0 - pad)
            ax[side].set_title(f"{name}", fontsize=13)
            ax[side].set_xlabel("um"); ax[side].grid(alpha=.2, lw=.5)
        ax[0].set_ylabel("um")
        med = t.dist_um.median() if "dist_um" in t else np.nan
        lab = (f"all {len(t)} numbered" if show_labels else
               f"{len(t)} pairs, too many to number legibly — "
               f"rerun with --tiles to split this region")
        ttl = f"{reg}" + (f", tile {int(tile)+1}" if a.tiles > 1 else "")
        fig.suptitle(f"{ttl} — {lab}. Same number is the same pair; "
                     f"median separation {med:.0f} um. Grey = unmatched.",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        suffix = f"_{int(tile)+1}" if a.tiles > 1 else ""
        f = f"{a.out}/matched_{reg}{suffix}.png"
        fig.savefig(f, dpi=150); plt.close(fig)
        t.to_csv(f"{a.out}/matched_{reg}{suffix}.csv", index=False)
        total_shown += len(t)
        rows.append(dict(region=reg, tile=int(tile) + 1, pairs=len(t),
                         labelled=show_labels, median_um=med, file=f))
        print(f"  {reg}{suffix}: {len(t):5d} pairs"
              f"{'' if show_labels else '  (labels omitted)'}  -> {f}")

S = pd.DataFrame(rows)
S.to_csv(f"{a.out}/index.csv", index=False)
print(f"\n  {total_shown:,} pairs drawn across {len(S)} figures")
print(f"  table has {len(P):,} pairs — these should agree")
if total_shown != len(P):
    print(f"  !! {len(P)-total_shown} missing: regions with fewer than two "
          f"pairs are skipped")
unlabelled = S[~S.labelled].pairs.sum()
if unlabelled:
    print(f"\n  {unlabelled:,} pairs are drawn but not numbered. Rerun with "
          f"--tiles 3 or 4 to label them all.")
print(f"\n  -> {a.out}/index.csv")
