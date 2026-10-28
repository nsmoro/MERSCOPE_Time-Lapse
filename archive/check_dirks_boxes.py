#!/usr/bin/env python
"""
Reproduce the marked-up view and extract the cells inside each box.

Mirrors the MERSCOPE patch horizontally, draws the boxes that were marked
on each panel, and pulls out the cells inside them. Then asks the two
questions that matter:

  1. Do the boxed regions hold a similar number of cells? If one box has
     12 cells and its partner has 40, they are not the same patch.

  2. Do the box pairs agree on an offset? Each pair implies one: the
     difference between the two box centres. If the boxes genuinely
     correspond, all of them imply nearly the same offset, because the
     whole dataset is displaced by a single amount. If they imply offsets
     hundreds of microns apart, they cannot all be right -- shifting the
     data to satisfy one moves it away from the others.

That second question is the useful one, and it is answerable from the
boxes alone without identifying individual cells.

Boxes are given as x0,y0,x1,y1 and may be repeated:

    python check_dirks_boxes.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --live-patch -13200 1750 -11600 3750 \
        --mer-patch 5700 1600 7500 3600 \
        --live-box -12950,2950,-12750,3200 \
        --live-box -12750,2700,-12450,3150 \
        --live-box -12450,3400,-12250,3650 \
        --mer-box 6250,2300,6450,2550 \
        --mer-box 6450,2150,6700,2500 \
        --mer-box 6700,2700,6900,2950 \
        --out ./dirk_boxes
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--live-patch", nargs=4, type=float, required=True)
ap.add_argument("--mer-patch", nargs=4, type=float, required=True)
ap.add_argument("--live-box", action="append", default=[])
ap.add_argument("--mer-box", action="append", default=[])
ap.add_argument("--mirror-mer", action="store_true", default=True,
                help="mirror the MERSCOPE patch horizontally (default on)")
ap.add_argument("--no-mirror-mer", dest="mirror_mer", action="store_false")
ap.add_argument("--out", default="./dirk_boxes")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"

def crop(P, b):
    x0,y0,x1,y1 = min(b[0],b[2]),min(b[1],b[3]),max(b[0],b[2]),max(b[1],b[3])
    return P[(P[:,0]>=x0)&(P[:,0]<=x1)&(P[:,1]>=y0)&(P[:,1]<=y1)]

A = crop(live[[lx,ly]].to_numpy(float), a.live_patch)
B = crop(mer[["center_x","center_y"]].to_numpy(float), a.mer_patch)
print(f"live patch     {len(A)} cells")
print(f"MERSCOPE patch {len(B)} cells")

if a.mirror_mer:
    lo, hi = B[:,0].min(), B[:,0].max()
    B = np.c_[lo + hi - B[:,0], B[:,1]]
    mbox = lambda b: [lo+hi-b[2], b[1], lo+hi-b[0], b[3]]
    print("  MERSCOPE patch mirrored horizontally")
else:
    mbox = lambda b: b

lb = [[float(v) for v in s.split(",")] for s in a.live_box]
mb = [mbox([float(v) for v in s.split(",")]) for s in a.mer_box]
if len(lb) != len(mb):
    raise SystemExit(f"{len(lb)} live boxes but {len(mb)} MERSCOPE boxes")
if not lb:
    raise SystemExit("give at least one --live-box and --mer-box")

print("\n" + "="*72)
print("CELLS INSIDE EACH BOX")
print("="*72)
print(f"{'box':>4} {'live n':>7} {'mer n':>7} {'ratio':>7}   "
      f"{'implied offset (um)':>24}")
print("-"*72)
rows = []
for k, (L_, M_) in enumerate(zip(lb, mb), 1):
    ca = crop(A, L_); cb = crop(B, M_)
    if len(ca) == 0 or len(cb) == 0:
        print(f"{k:>4} {len(ca):>7} {len(cb):>7}    box is empty")
        continue
    off = cb.mean(0) - ca.mean(0)
    rows.append(dict(box=k, n_live=len(ca), n_mer=len(cb),
                     dx=off[0], dy=off[1]))
    print(f"{k:>4} {len(ca):>7} {len(cb):>7} {len(cb)/len(ca):>7.2f}   "
          f"({off[0]:>+10.1f},{off[1]:>+10.1f})")
    pd.DataFrame(ca, columns=["x_um","y_um"]).to_csv(
        f"{a.out}/box{k}_live_cells.csv", index=False)
    pd.DataFrame(cb, columns=["x_um","y_um"]).to_csv(
        f"{a.out}/box{k}_mer_cells.csv", index=False)

R = pd.DataFrame(rows)
if len(R) > 1:
    print("\n" + "="*72); print("DO THE BOXES AGREE?"); print("="*72)
    print(f"  mean offset   ({R.dx.mean():+.1f}, {R.dy.mean():+.1f}) um")
    print(f"  spread        dx {R.dx.std():.1f}, dy {R.dy.std():.1f} um")
    sd = np.hypot(R.dx.std(), R.dy.std())
    print(f"  combined      {sd:.1f} um\n")
    if sd < 20:
        print("  The boxes agree on one offset to within a cell spacing.")
        print("  That offset is the transform - apply it to everything.")
    elif sd < 80:
        print("  Roughly consistent, but scattered by more than a cell")
        print("  spacing. Worth identifying individual cells to tighten it.")
    else:
        print("  The boxes imply offsets that differ by much more than the")
        print("  distance between cells, so they cannot all be describing")
        print("  the same shift. Moving the data to line up one box moves")
        print("  it away from the others.")
R.to_csv(f"{a.out}/box_offsets.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, ax = plt.subplots(1, 2, figsize=(18, 9))
for k, (P, boxes, name, c) in enumerate([(A, lb, "LIVE", "#1D9E75"),
        (B, mb, "MERSCOPE" + (" (mirrored)" if a.mirror_mer else ""), "#B03A8B")]):
    ax[k].scatter(P[:,0], P[:,1], s=16, c=c, zorder=2)
    for i, b in enumerate(boxes, 1):
        x0,y0 = min(b[0],b[2]), min(b[1],b[3])
        ax[k].add_patch(Rectangle((x0,y0), abs(b[2]-b[0]), abs(b[3]-b[1]),
                                  fill=False, ec="#111", lw=2, zorder=4))
        ax[k].annotate(str(i), (x0, y0), fontsize=14, weight="bold",
                       color="#111", va="bottom", ha="right", zorder=5)
        inside = crop(P, b)
        ax[k].scatter(inside[:,0], inside[:,1], s=34, c="#C0392B", zorder=3)
    ax[k].set_aspect("equal"); ax[k].invert_yaxis()
    ax[k].set_title(f"{name} — {len(P)} cells", fontsize=13)
    ax[k].set_xlabel("um")
ax[0].set_ylabel("um")
fig.suptitle("The marked boxes, with the cells inside them highlighted",
             fontsize=15)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f"{a.out}/boxes_marked.png", dpi=150)
print(f"\n  -> {a.out}/boxes_marked.png")
print(f"  -> {a.out}/box_offsets.csv  and box*_live_cells.csv / box*_mer_cells.csv")
