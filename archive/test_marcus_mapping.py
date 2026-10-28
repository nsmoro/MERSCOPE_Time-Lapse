#!/usr/bin/env python
"""
Test the field-of-view to region mapping supplied by whoever ran the slide.

Every mapping tested so far was inferred from the geometry. This one is
ground truth: each MERSCOPE region corresponds to a named group of
microscope positions, stated by the person who drew the regions.

Default grouping is three consecutive positions per region, in order --
p0001-p0003 to the first region, p0004-p0006 to the second, and so on,
with any remaining positions going to the last region. Override with
--groups if the grouping differs.

Each pair is tested on its own: is there one shift at which those live
cells land on those MERSCOPE cells? Several pairs agreeing on a shift
cannot happen by chance.

    python test_marcus_mapping.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region-order R9,R11,R12,R13,R14,R16,R17,R18,R20 \
        --out ./marcus
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_correspondence import search_rotation, empirical_null

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region-order", required=True,
                help="regions in the order they correspond to the positions")
ap.add_argument("--per-region", type=int, default=3,
                help="how many consecutive positions per region")
ap.add_argument("--groups", default=None,
                help="explicit, e.g. R9:1-3,R11:4-6,R20:25-36")
ap.add_argument("--flip", default="all",
                help="all | none | x | y | xy — which orientations to try")
ap.add_argument("--max-shift", type=float, default=1500.)
ap.add_argument("--bin-um", type=float, default=5.)
ap.add_argument("--rot-range", type=float, default=4.)
ap.add_argument("--rot-step", type=float, default=0.5)
ap.add_argument("--out", default="./marcus")
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

print("MAPPING UNDER TEST")
for reg, ps in groups.items():
    print(f"  {reg:>5}  <-  positions {ps[0]}..{ps[-1]}  ({len(ps)} FOVs)")

FLIPS = ([("none",1,1),("x",-1,1),("y",1,-1),("xy",-1,-1)]
         if a.flip == "all" else
         [(a.flip, -1 if "x" in a.flip else 1, -1 if "y" in a.flip else 1)])
ang = np.arange(-a.rot_range, a.rot_range+1e-9, a.rot_step)

print(f"\n{'region':>7} {'FOVs':>12} {'n_live':>7} {'n_mer':>7} "
      f"{'flip':>5} {'peak z':>8} {'sigma':>7}   {'offset (um)':>22}")
print("-"*86)
rows = []
for reg, ps in groups.items():
    sub = live[live.Position.isin(ps)]
    B = mer[mer.region == reg][["center_x","center_y"]].to_numpy(float)
    if len(sub) < 60 or len(B) < 60:
        print(f"{reg:>7} {f'{ps[0]}-{ps[-1]}':>12} {len(sub):>7} {len(B):>7}"
              f"   too few cells")
        continue
    A0 = sub[[lx, ly]].to_numpy(float)
    best = None
    for fname, sx, sy in FLIPS:
        A = A0 * np.array([sx, sy], float)
        pre = B.mean(0) - A.mean(0)
        bb, _ = search_rotation(A + pre, B, ang, a.max_shift, a.bin_um,
                                verbose=False)
        pz, th, dx, dy, z, _ = bb
        nm, ns, _ = empirical_null(A + pre, B, ang, a.max_shift, a.bin_um,
                                   n_draws=3)
        sig = (pz - nm) / ns
        if best is None or sig > best[0]:
            best = (sig, fname, pz, th, dx+pre[0], dy+pre[1])
    sig, fname, pz, th, dx, dy = best
    rows.append(dict(region=reg, fovs=f"{ps[0]}-{ps[-1]}", n_live=len(A0),
                     n_mer=len(B), flip=fname, z=pz, sigma=sig,
                     angle=th, dx=dx, dy=dy))
    print(f"{reg:>7} {f'{ps[0]}-{ps[-1]}':>12} {len(A0):>7} {len(B):>7} "
          f"{fname:>5} {pz:>8.2f} {sig:>+7.1f}   ({dx:>+9.0f},{dy:>+9.0f})")

if not rows:
    raise SystemExit("\nno region had enough cells")
R = pd.DataFrame(rows); R.to_csv(f"{a.out}/marcus_mapping.csv", index=False)

print("\n" + "="*72); print("DOES IT HOLD?"); print("="*72)
good = R[R.sigma > 3]
print(f"  {len(good)}/{len(R)} regions above 3 sigma")
if len(good):
    print(f"  orientations chosen: {dict(good.flip.value_counts())}")
if len(good) >= 2:
    print(f"  offsets among them: dx spread {good.dx.std():.0f} um, "
          f"dy spread {good.dy.std():.0f} um")
    if good.dx.std() < 150 and good.dy.std() < 150:
        print("\n  Those regions agree on a common shift. That is a real")
        print("  alignment -- use this mapping and match within each pair.")
    else:
        print("\n  They score well individually but disagree on the shift,")
        print("  so they cannot all describe the same transform.")
elif len(good) == 1:
    print("\n  One region scoring on its own could be chance. Two or more")
    print("  agreeing is what cannot be.")
else:
    print("\n  No region shows a cell-level alignment under this mapping,")
    print("  even though the mapping itself comes from the bench.")
print(f"\n  -> {a.out}/marcus_mapping.csv")

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

cmap = plt.get_cmap("tab10")
colour = {reg: cmap(k % 10) for k, reg in enumerate(groups)}

# 1. the mapping itself: both datasets coloured by the pairing
fig, ax = plt.subplots(1, 2, figsize=(21, 9))
for reg, ps in groups.items():
    sub = live[live.Position.isin(ps)]
    ax[0].scatter(sub[lx], sub[ly], s=1.4, color=colour[reg], linewidths=0)
    c = (sub[lx].mean(), sub[ly].mean())
    ax[0].annotate(f"{reg}\n{ps[0]}-{ps[-1]}", c, fontsize=10, weight="bold",
                   ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=.85))
    g = mer[mer.region == reg]
    if len(g):
        ax[1].scatter(g.center_x, g.center_y, s=1.4, color=colour[reg],
                      linewidths=0)
        ax[1].annotate(reg, (g.center_x.mean(), g.center_y.mean()),
                       fontsize=11, weight="bold", ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                 alpha=.85))
for A, t in zip(ax, ["TIME-LAPSE, positions grouped by region",
                     "MERSCOPE, cells by region"]):
    A.set_aspect("equal"); A.invert_yaxis(); A.set_title(t, fontsize=13)
    A.set_xlabel("um")
ax[0].set_ylabel("um")
fig.suptitle("The mapping under test — matching colours should be the same cells",
             fontsize=15)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(f"{a.out}/mapping_layout.png", dpi=145); plt.close(fig)
print(f"  -> {a.out}/mapping_layout.png")

# 2. one panel per region pair, live beside MERSCOPE
n = len(rows)
fig, ax = plt.subplots(2, n, figsize=(3.1*n, 7.6), squeeze=False)
for k, r in enumerate(rows):
    reg = r["region"]; ps = groups[reg]
    sub = live[live.Position.isin(ps)]
    A0 = sub[[lx, ly]].to_numpy(float)
    sx = -1 if "x" in r["flip"] else 1
    sy = -1 if "y" in r["flip"] else 1
    A0 = A0 * np.array([sx, sy], float)
    B = mer[mer.region == reg][["center_x","center_y"]].to_numpy(float)
    for row, (P, lab, c) in enumerate([(A0, f"live {ps[0]}-{ps[-1]}", "#1D9E75"),
                                       (B, f"MERSCOPE {reg}", "#B03A8B")]):
        A = ax[row][k]
        A.scatter(P[:,0]-P[:,0].min(), P[:,1]-P[:,1].min(), s=1.2, c=c,
                  linewidths=0)
        A.set_aspect("equal"); A.invert_yaxis()
        A.set_xticks([]); A.set_yticks([])
        A.set_title(lab if row else
                    f"{lab}\n{r['flip']}  {r['sigma']:+.1f}\u03c3",
                    fontsize=9,
                    color=("#1A7A4C" if r["sigma"] > 3 else "#A3281E")
                          if row == 0 else "black")
fig.suptitle("Each pair under test, live above and MERSCOPE below "
             "(origins removed, so only the shape matters)", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig(f"{a.out}/region_pairs.png", dpi=145); plt.close(fig)
print(f"  -> {a.out}/region_pairs.png")
