#!/usr/bin/env python
"""
Side-by-side plot with the live data pre-oriented to match MERSCOPE.

The registration already found that the live view is related to the
MERSCOPE view by flip_xy -- a 180 degree rotation. This applies that
orientation before plotting, so the two panels can be compared directly
by eye without mentally rotating one of them.

    python plot_oriented.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv --transform rot180 \
        --out ./oriented

--transform options:
    none       as-is
    rot180     180 degree rotation  (= flip both axes; what the fit found)
    mirror-x   mirror left-right
    mirror-y   mirror top-bottom
    rot180-mx  180 degrees then mirror left-right
    rot90      quarter turn
    rot270     quarter turn the other way
"""
import argparse, os
import numpy as np, pandas as pd

T = {
    "none":      lambda p: p,
    "rot180":    lambda p: p * [-1, -1],
    "mirror-x":  lambda p: p * [-1,  1],
    "mirror-y":  lambda p: p * [ 1, -1],
    "rot180-mx": lambda p: p * [-1, -1] * [-1, 1],
    "rot90":     lambda p: np.c_[-p[:, 1],  p[:, 0]],
    "rot270":    lambda p: np.c_[ p[:, 1], -p[:, 0]],
}

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--transform", default="rot180", choices=list(T))
ap.add_argument("--out", default="./oriented")
ap.add_argument("--point-size", type=float, default=1.5)
ap.add_argument("--all", action="store_true",
                help="draw every transform, to compare them at once")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
L = live[[lx, ly]].to_numpy(float)
M = mer[["center_x", "center_y"]].to_numpy(float)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def panel(ax, P, title, colour):
    Q = P - P.min(0)                      # drop the arbitrary origin
    ax.scatter(Q[:, 0], Q[:, 1], s=a.point_size, c=colour, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{title}\n{np.ptp(Q,0)[0]:.0f} x {np.ptp(Q,0)[1]:.0f} um",
                 fontsize=12)
    ax.set_xlabel("um"); ax.set_ylabel("um")

if a.all:
    keys = list(T)
    fig, ax = plt.subplots(2, 4, figsize=(22, 11))
    for k, key in enumerate(keys):
        panel(ax.flat[k], T[key](L), f"live: {key}", "#1D9E75")
    panel(ax.flat[len(keys)], M, "MERSCOPE", "#B03A8B")
    for j in range(len(keys) + 1, 8):
        ax.flat[j].axis("off")
    fig.suptitle("Live data under each orientation, with MERSCOPE for comparison",
                 fontsize=15)
    fig.tight_layout()
    f = f"{a.out}/orientations_all.png"
else:
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    panel(ax[0], T[a.transform](L), f"Live ({len(L):,} cells, {a.transform})", "#1D9E75")
    panel(ax[1], M, f"MERSCOPE ({len(M):,} cells)", "#B03A8B")
    fig.suptitle("Same scale, same orientation — compare directly", fontsize=14)
    fig.tight_layout()
    f = f"{a.out}/oriented_side_by_side.png"

fig.savefig(f, dpi=150); print(f"-> {f}")
print("\nBoth panels start at (0,0) so only the shape matters, not the")
print("absolute coordinates. If the arrangement of blocks corresponds,")
print("the orientation is right.")
