#!/usr/bin/env python
"""
Every cell, tile by tile, live beside MERSCOPE, with nothing highlighted.

No match colouring and no numbering -- just the two cell distributions
side by side at matched scale, cut into tiles, with a locator showing
which part of the slide each tile covers. For looking at the patterns
directly rather than at what an algorithm decided.

Both panels of a tile cover the same fraction of their region, and each
pair of panels is drawn on its own axes with equal aspect, so shapes are
comparable between them.

    python tiles_plain.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region R18 --positions 22,23,24 \
        --rotate merscope --tiles 4 --out ./plain_tiles
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region", default=None)
ap.add_argument("--positions", default=None)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--all", action="store_true",
                help="one figure per region, using the standard grouping")
ap.add_argument("--rotate", default="merscope",
                choices=["merscope", "live", "none"])
ap.add_argument("--flip", default="none", choices=["none","x","y","xy"],
                help="global flip of the live coordinates")
ap.add_argument("--mirror", default="none", choices=["none","x","y","xy"],
                help="mirror each region block about its own centre")
ap.add_argument("--reorder", default="none", choices=["none","x","y"],
                help="reverse the order of the fields within each region")
ap.add_argument("--save-csv", action="store_true",
                help="also write the transformed coordinates")
ap.add_argument("--tiles", type=int, default=4)
ap.add_argument("--point-size", type=float, default=9.0)
ap.add_argument("--out", default="./plain_tiles")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"

regions = [r.strip() for r in a.region_order.split(",")]
positions = sorted(live.Position.unique())
groups = {}
for k, reg in enumerate(regions):
    lo = k * a.per_region
    hi = (k + 1) * a.per_region if k < len(regions) - 1 else len(positions)
    groups[reg] = positions[lo:hi]

if a.all:
    jobs = [(r, p) for r, p in groups.items()]
else:
    if not a.region:
        raise SystemExit("give --region (and --positions), or --all")
    ps = ([int(p) for p in a.positions.split(",")] if a.positions
          else groups.get(a.region, []))
    jobs = [(a.region, ps)]

def rot180(P):
    lo, hi = P.min(0), P.max(0)
    return (lo + hi) - P

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

TEAL, PINK = "#1D9E75", "#B03A8B"
B_whole = mer[["center_x", "center_y"]].to_numpy(float)

for reg, ps in jobs:
    sub = live[live.Position.isin(ps)]
    A = sub[[lx, ly]].to_numpy(float)
    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    if len(A) < 10 or len(B) < 10:
        print(f"{reg}: too few cells"); continue

    if a.flip != "none":
        A = A * np.array([-1 if "x" in a.flip else 1,
                          -1 if "y" in a.flip else 1], float)
    if a.mirror != "none":
        if "x" in a.mirror:
            c = (A[:,0].min() + A[:,0].max())/2; A[:,0] = 2*c - A[:,0]
        if "y" in a.mirror:
            c = (A[:,1].min() + A[:,1].max())/2; A[:,1] = 2*c - A[:,1]
    if a.reorder != "none":
        ai = 0 if a.reorder == "x" else 1
        pos = sub.Position.to_numpy()
        cen = {q: A[pos == q].mean(0) for q in np.unique(pos)}
        order = sorted(cen, key=lambda q: cen[q][ai])
        for src, dst in zip(order, order[::-1]):
            A[pos == src] += cen[dst] - cen[src]
    if a.rotate == "merscope":
        B = rot180(B)
    elif a.rotate == "live":
        A = rot180(A)

    def frac(P):
        lo, hi = P.min(0), P.max(0)
        return (P - lo) / np.maximum(hi - lo, 1e-9)
    fA, fB = frac(A), frac(B)

    nt = a.tiles
    tiles = []
    for ty in range(nt):
        for tx in range(nt):
            lo = np.array([tx, ty]) / nt
            hi = np.array([tx + 1, ty + 1]) / nt
            sa = np.all((fA >= lo) & (fA < hi), axis=1)
            sb = np.all((fB >= lo) & (fB < hi), axis=1)
            if sa.sum() >= 3 and sb.sum() >= 3:
                tiles.append((ty * nt + tx + 1, A[sa], B[sb], lo, hi))

    ncol = min(4, len(tiles))
    nrow = int(np.ceil(len(tiles) / ncol))
    fig = plt.figure(figsize=(5.4 * ncol, 3.8 + 4.4 * nrow))
    gs = GridSpec(nrow + 1, ncol * 2, figure=fig,
                  height_ratios=[0.8] + [1] * nrow, hspace=0.34, wspace=0.16)

    ax = fig.add_subplot(gs[0, :ncol])
    ax.scatter(B_whole[:, 0], B_whole[:, 1], s=.5, c="#DDE2E6", linewidths=0)
    gm = mer[mer.region == reg]
    ax.scatter(gm.center_x, gm.center_y, s=.7, c=PINK, linewidths=0)
    ax.add_patch(Rectangle((gm.center_x.min(), gm.center_y.min()),
                           np.ptp(gm.center_x), np.ptp(gm.center_y),
                           fill=False, ec="#1D6F8C", lw=2.2))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"whole slide — {reg} outlined", fontsize=11)

    ax = fig.add_subplot(gs[0, ncol:])
    ax.scatter(B[:, 0], B[:, 1], s=1.2, c=PINK, linewidths=0)
    for t, _, Bt, lo, hi in tiles:
        x0 = B[:, 0].min() + lo[0] * np.ptp(B[:, 0])
        y0 = B[:, 1].min() + lo[1] * np.ptp(B[:, 1])
        ax.add_patch(Rectangle((x0, y0), np.ptp(B[:, 0]) / nt,
                               np.ptp(B[:, 1]) / nt, fill=False,
                               ec="#1D6F8C", lw=1.1))
        ax.annotate(str(t), (x0 + np.ptp(B[:, 0]) / nt / 2,
                             y0 + np.ptp(B[:, 1]) / nt / 2),
                    fontsize=10, weight="bold", color="#1D6F8C",
                    ha="center", va="center")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{reg} in {len(tiles)} tiles", fontsize=11)

    for k, (t, At, Bt, lo, hi) in enumerate(tiles):
        r, c = divmod(k, ncol)
        for side, (P, colour, nm) in enumerate([(At, TEAL, "live"),
                                                (Bt, PINK, "MERSCOPE")]):
            axp = fig.add_subplot(gs[r + 1, c * 2 + side])
            Q = P - (P.min(0) + P.max(0)) / 2
            axp.scatter(Q[:, 0], Q[:, 1], s=a.point_size, c=colour,
                        linewidths=0)
            axp.set_aspect("equal"); axp.invert_yaxis()
            axp.set_xticks([]); axp.set_yticks([])
            axp.set_title(f"tile {t} · {nm}" +
                          (f"  ({len(P)} cells)" if side == 0 else
                           f"  ({len(P)} cells)"), fontsize=9.5)

    fig.suptitle(f"{reg} — every cell, {len(A)} live and {len(B)} MERSCOPE"
                 + (f", MERSCOPE rotated 180" if a.rotate == "merscope" else
                    f", live rotated 180" if a.rotate == "live" else "")
                 + (f", live flipped {a.flip}" if a.flip != "none" else "")
                 + (f", blocks mirrored {a.mirror}" if a.mirror != "none" else "")
                 + (f", fields swapped on {a.reorder}" if a.reorder != "none" else ""),
                 fontsize=14, y=0.995)
    if a.save_csv:
        pd.DataFrame(A, columns=["x_oriented","y_oriented"]).assign(
            Position=sub.Position.to_numpy()).to_csv(
            f"{a.out}/{reg}_live_oriented.csv", index=False)
        pd.DataFrame(B, columns=["x_oriented","y_oriented"]).to_csv(
            f"{a.out}/{reg}_merscope_oriented.csv", index=False)
    f = f"{a.out}/{reg}_plain.png"
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  {reg}: {len(A)} live, {len(B)} MERSCOPE, {len(tiles)} tiles "
          f"-> {f}")

print("\n  Each tile covers the same fraction of the region on both sides,")
print("  and both panels are centred, so only the arrangement differs.")
