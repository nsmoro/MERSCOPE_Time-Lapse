#!/usr/bin/env python
"""
Mirror the MERSCOPE coordinates and plot them beside the live data.

Applies a horizontal flip to the MERSCOPE x-axis, leaving the live data as
the reference frame. Cell identities are untouched -- only coordinates
change -- so EntityID still joins to cell_by_gene.csv afterwards.

Note this is the same handedness change as mirroring the live x-axis
instead; which side you flip is a matter of which frame you treat as the
reference. Doing it on the MERSCOPE side keeps the live data, with its
tracks and lineages, in its own untouched coordinates.

    python flip_merscope.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv --axis x --out ./flipped

Then register against the flipped file:

    python register_and_match.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope ./flipped/merscope_flipx.csv --out ./flipped/match
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--axis", default="x", choices=["x", "y", "xy"],
                help="which axis to mirror; x = horizontal flip")
ap.add_argument("--out", default="./flipped")
ap.add_argument("--point-size", type=float, default=1.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"

m = mer.copy()
if "x" in a.axis:
    m["center_x"] = m.center_x.max() + m.center_x.min() - m.center_x
if "y" in a.axis:
    m["center_y"] = m.center_y.max() + m.center_y.min() - m.center_y
# mirror about the midpoint so the data stays in the same coordinate range
out_csv = f"{a.out}/merscope_flip{a.axis}.csv"
m.to_csv(out_csv, index=False)
print(f"mirrored MERSCOPE on {a.axis}  ->  {out_csv}")
print(f"  {len(m):,} cells, extent unchanged: "
      f"{np.ptp(m.center_x):.0f} x {np.ptp(m.center_y):.0f} um")
print("  EntityIDs untouched, so cell_by_gene.csv still joins")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def panel(ax, X, Y, title, colour, ps):
    ax.scatter(X, Y, s=ps, c=colour, linewidths=0)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("um"); ax.set_ylabel("um")

fig, ax = plt.subplots(1, 3, figsize=(24, 8))
panel(ax[0], live[lx], live[ly],
      f"TIME-LAPSE (reference)\n{len(live):,} cells", "#1D9E75", a.point_size)
panel(ax[1], mer.center_x, mer.center_y,
      f"MERSCOPE (as acquired)\n{len(mer):,} cells", "#B03A8B", a.point_size)
panel(ax[2], m.center_x, m.center_y,
      f"MERSCOPE ({a.axis}-mirrored)\n{len(m):,} cells", "#7A3D96", a.point_size)
fig.suptitle("Time-lapse as reference, MERSCOPE before and after the mirror",
             fontsize=15, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
f = f"{a.out}/side_by_side_flipped.png"
fig.savefig(f, dpi=150); plt.close(fig)
print(f"  -> {f}")

# per-region, since the regions are what get compared
if "region" in mer:
    regs = sorted(mer.region.unique())
    n = len(regs)
    fig, ax = plt.subplots(2, n, figsize=(2.9*n, 7.5), squeeze=False)
    for k, r in enumerate(regs):
        g0 = mer[mer.region == r]; g1 = m[m.region == r]
        for row, (g, lab, c) in enumerate([(g0, "as acquired", "#B03A8B"),
                                           (g1, f"{a.axis}-mirrored", "#7A3D96")]):
            A = ax[row][k]
            A.scatter(g.center_x - g.center_x.min(),
                      g.center_y - g.center_y.min(), s=1, c=c, linewidths=0)
            A.set_aspect("equal"); A.invert_yaxis()
            A.set_xticks([]); A.set_yticks([])
            A.set_title(f"{r}\n{lab}" if row == 0 else lab, fontsize=9)
    fig.suptitle("Each MERSCOPE region, before and after the mirror", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    f = f"{a.out}/regions_flipped.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"  -> {f}")

print(f"\nNext: register against the mirrored file\n"
      f"  python register_and_match.py \\\n"
      f"    --live {a.live} \\\n"
      f"    --merscope {out_csv} \\\n"
      f"    --out {a.out}/match")
