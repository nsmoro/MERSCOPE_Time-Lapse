#!/usr/bin/env python
"""
Which orientation is consistent with a landmark pair?

Given one feature identified in both datasets -- a cluster, a gap, a
distinctive patch -- this asks which of the eight orientations actually
puts the live feature where the MERSCOPE feature is.

This is a weaker test than matching individual cells, but it settles a
different and prior question: which orientation is even worth testing. A
landmark in column 5 of the bottom band cannot correspond to one in
column 2 of the top band under a horizontal flip, because a horizontal
flip leaves the band alone. It can under a 180 degree rotation, which
flips both.

    python check_orientation.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --live-point -12500 3000 --mer-point 6500 2700
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--live-point", nargs=2, type=float, required=True)
ap.add_argument("--mer-point", nargs=2, type=float, required=True)
a = ap.parse_args()

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
L = live[[lx, ly]].to_numpy(float)
M = mer[["center_x", "center_y"]].to_numpy(float)
P = np.array(a.live_point, float)
Q = np.array(a.mer_point, float)

def frac(p, D):
    """where a point sits inside its dataset, as a fraction 0-1"""
    lo, hi = D.min(0), D.max(0)
    return (p - lo) / (hi - lo)

fp, fq = frac(P, L), frac(Q, M)
print(f"live landmark  sits at {fp[0]:.2f} across, {fp[1]:.2f} down")
print(f"MERSCOPE       sits at {fq[0]:.2f} across, {fq[1]:.2f} down")
print(f"\n  (0 = left/top edge, 1 = right/bottom edge of each dataset)\n")

OPS = {
    "none":            (lambda f: f),
    "flip x":          (lambda f: np.array([1-f[0], f[1]])),
    "flip y":          (lambda f: np.array([f[0], 1-f[1]])),
    "rot180 (x and y)":(lambda f: np.array([1-f[0], 1-f[1]])),
    "rot90":           (lambda f: np.array([1-f[1], f[0]])),
    "rot270":          (lambda f: np.array([f[1], 1-f[0]])),
    "transpose":       (lambda f: np.array([f[1], f[0]])),
    "anti-transpose":  (lambda f: np.array([1-f[1], 1-f[0]])),
}

print(f"{'orientation':<20} {'live lands at':<18} {'error':>8}")
print("-"*50)
rows = []
for name, op in OPS.items():
    g = op(fp)
    err = np.hypot(*(g - fq))
    rows.append((err, name, g))
    print(f"{name:<20} {g[0]:.2f} across, {g[1]:.2f} down   {err:>6.3f}")

rows.sort()
err, name, g = rows[0]
print(f"\n  best fit: {name}  (error {err:.3f})")
print(f"  next:     {rows[1][1]}  (error {rows[1][0]:.3f})")
print()
if err < 0.08:
    print(f"  Only {name} puts the live landmark where the MERSCOPE one is.")
    if rows[1][0] - err > 0.15:
        print("  The alternatives are clearly worse, so this is well separated.")
else:
    print("  No orientation places them together. Either the two landmarks")
    print("  are not the same feature, or the datasets do not overlap the")
    print("  way we assume.")
print("\n  This only fixes the orientation. Whether the individual cells")
print("  inside those landmarks correspond is a separate question.")
