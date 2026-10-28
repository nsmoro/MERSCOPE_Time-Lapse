#!/usr/bin/env python
"""
Overlay one live region on its MERSCOPE partner, drawn on one set of axes.

Three panels: the two separately at matched scale, then both on top of
each other. Equal aspect everywhere, so a square region looks square and
any apparent rotation is real rather than an artefact of the axes being
stretched differently.

The live points are oriented, then shifted so the two blocks share a
centre. That removes the arbitrary offset between coordinate systems and
leaves only the question that matters: do the two point clouds have the
same shape and the same internal structure?

    python overlay_regions.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --positions 22,23,24 --region R18 --flip xy --out ./overlay_R18

    # every region in one go
    python overlay_regions.py ... --all --out ./overlay_all
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--positions", default=None, help="e.g. 22,23,24")
ap.add_argument("--region", default=None, help="e.g. R18")
ap.add_argument("--all", action="store_true",
                help="draw every region using the standard grouping")
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy")
ap.add_argument("--local-flip", default="none",
                help="none|x|y|xy — applied to each region about its OWN "
                     "centre, after the global flip. Different from --flip: "
                     "the global one moves blocks to new places, this one "
                     "leaves each block where it is and reverses the cells "
                     "inside it.")
ap.add_argument("--point-size", type=float, default=7.0)
ap.add_argument("--out", default="./overlay")
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

if a.all:
    pairs = [(reg, ps) for reg, ps in groups.items()]
else:
    if not (a.positions and a.region):
        raise SystemExit("give --positions and --region, or use --all")
    pairs = [(a.region, [int(p) for p in a.positions.split(",")])]

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def centred(P):
    return P - (P.min(0) + P.max(0)) / 2

for reg, ps in pairs:
    A = live[live.Position.isin(ps)][[lx, ly]].to_numpy(float) * [sx, sy]
    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    if len(A) < 10 or len(B) < 10:
        print(f"{reg}: too few cells ({len(A)} live, {len(B)} MERSCOPE)")
        continue
    Ac, Bc = centred(A), centred(B)
    if a.local_flip != "none":
        if "x" in a.local_flip: Ac[:, 0] = -Ac[:, 0]
        if "y" in a.local_flip: Ac[:, 1] = -Ac[:, 1]

    print(f"{reg}  positions {ps[0]}-{ps[-1]}")
    print(f"   live     {len(A):5d} cells   {np.ptp(A,0)[0]:6.0f} x "
          f"{np.ptp(A,0)[1]:6.0f} um   ratio {np.ptp(A,0)[0]/np.ptp(A,0)[1]:.2f}")
    print(f"   MERSCOPE {len(B):5d} cells   {np.ptp(B,0)[0]:6.0f} x "
          f"{np.ptp(B,0)[1]:6.0f} um   ratio {np.ptp(B,0)[0]/np.ptp(B,0)[1]:.2f}")

    lim = max(np.abs(Ac).max(), np.abs(Bc).max()) * 1.05
    fig, ax = plt.subplots(1, 3, figsize=(19, 7.5))
    ax[0].scatter(Ac[:,0], Ac[:,1], s=a.point_size, c="#1D9E75", linewidths=0)
    ax[0].set_title(f"live, positions {ps[0]}-{ps[-1]}  ({len(A)} cells)",
                    fontsize=12)
    ax[1].scatter(Bc[:,0], Bc[:,1], s=a.point_size, c="#B03A8B", linewidths=0)
    ax[1].set_title(f"MERSCOPE {reg}  ({len(B)} cells)", fontsize=12)
    ax[2].scatter(Bc[:,0], Bc[:,1], s=a.point_size*1.3, c="#B03A8B",
                  linewidths=0, alpha=.55, label="MERSCOPE")
    ax[2].scatter(Ac[:,0], Ac[:,1], s=a.point_size*1.3, c="#1D9E75",
                  linewidths=0, alpha=.55, label="live")
    ax[2].set_title("overlaid, centres aligned", fontsize=12)
    ax[2].legend(markerscale=2.2, fontsize=10)
    for A_ in ax:
        A_.set_aspect("equal"); A_.invert_yaxis()
        A_.set_xlim(-lim, lim); A_.set_ylim(lim, -lim)
        A_.set_xlabel("um (centred)"); A_.grid(alpha=.2, lw=.5)
    ax[0].set_ylabel("um (centred)")
    tag = (f"   [each region flipped '{a.local_flip}' about its own centre]"
           if a.local_flip != "none" else "")
    fig.suptitle(f"{reg} — equal aspect on all three, so any rotation is "
                 f"real{tag}", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.94])
    f = f"{a.out}/overlay_{reg}.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(Bc).query(Ac, k=1)
    print(f"   after centring{' and local flip' if a.local_flip!='none' else ''}: "
          f"median nearest-neighbour {np.median(d):.1f} um, "
          f"{(d<10).mean():.1%} within 10 um")
    print(f"   -> {f}\n")

print("In the third panel, the two blocks should cover the same footprint.")
print("If one is clearly rotated relative to the other, that is real - the")
print("axes are equal and both are centred, so nothing is being stretched.")
