#!/usr/bin/env python
"""
Test a proposed well-to-well mapping, one pair at a time.

Divides each dataset into a grid of wells (numbered left to right, then
top to bottom), applies a proposed mapping between them, and runs the
displacement test on every pair independently.

The advantage over a global fit is that each well pair is judged on its
own. If the mapping is right, most pairs should show a peak and they
should agree on a common offset. If it is wrong, they will scatter -- and
you can see which specific pairs fail rather than only that "it didn't
work".

    # default mapping swaps the two bands: 1<->7, 2<->8, ...
    python test_well_mapping.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv --mapping band-swap

    # or give it explicitly, live:merscope
    python test_well_mapping.py ... --mapping 1:7,2:8,3:9,4:10,5:11,6:12
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_correspondence import displacement_spectrum, search_rotation, empirical_null

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--ncols", type=int, default=6)
ap.add_argument("--nbands", type=int, default=2)
ap.add_argument("--mapping", default="band-swap")
ap.add_argument("--flip", default="none", help="none|x|y|xy applied to live")
ap.add_argument("--max-shift", type=float, default=1200.)
ap.add_argument("--bin-um", type=float, default=5.)
ap.add_argument("--rot-range", type=float, default=3.)
ap.add_argument("--rot-step", type=float, default=0.5)
ap.add_argument("--out", default="./wells")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def wells(P, ncols, nbands):
    """Split a point set into ncols x nbands wells by the gaps between them."""
    def cut(v, n):
        v = np.sort(v)
        gaps = np.diff(v)
        idx = np.argsort(gaps)[-(n-1):] if n > 1 else []
        edges = sorted([v[0]-1] + [(v[i]+v[i+1])/2 for i in idx] + [v[-1]+1])
        return edges
    ex, ey = cut(P[:,0], ncols), cut(P[:,1], nbands)
    col = np.clip(np.digitize(P[:,0], ex[1:-1]), 0, ncols-1)
    band = np.clip(np.digitize(P[:,1], ey[1:-1]), 0, nbands-1)
    return band*ncols + col + 1          # 1..ncols*nbands, left-right then top-bottom

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
L = live[[lx, ly]].to_numpy(float)
M = mer[["center_x","center_y"]].to_numpy(float)
sx = -1 if "x" in a.flip else 1
sy = -1 if "y" in a.flip else 1
L = L * np.array([sx, sy], float)

wl = wells(L, a.ncols, a.nbands)
wm = wells(M, a.ncols, a.nbands)
N = a.ncols * a.nbands
print(f"live wells:     {dict(pd.Series(wl).value_counts().sort_index())}")
print(f"MERSCOPE wells: {dict(pd.Series(wm).value_counts().sort_index())}\n")

if a.mapping == "band-swap":
    mapping = {i: (i + a.ncols - 1) % N + 1 for i in range(1, N+1)}
elif a.mapping == "identity":
    mapping = {i: i for i in range(1, N+1)}
else:
    mapping = {int(k): int(v) for k, v in
               (p.split(":") for p in a.mapping.split(","))}
print(f"testing mapping: {mapping}\n")

ang = np.arange(-a.rot_range, a.rot_range+1e-9, a.rot_step)
rows = []
print(f"{'live':>5} {'mer':>5} {'n_live':>7} {'n_mer':>7} {'peak z':>8} "
      f"{'sigma':>7}  {'offset (um)':>20}")
print("-"*70)
for lw, mw in mapping.items():
    A = L[wl == lw]; B = M[wm == mw]
    if len(A) < 60 or len(B) < 60:
        print(f"{lw:>5} {mw:>5} {len(A):>7} {len(B):>7}   too few cells")
        continue
    pre = B.mean(0) - A.mean(0)
    best, _ = search_rotation(A + pre, B, ang, a.max_shift, a.bin_um, verbose=False)
    pz, th, dx, dy, z, _ = best
    nm, ns, _ = empirical_null(A + pre, B, ang, a.max_shift, a.bin_um, n_draws=3)
    sig = (pz - nm) / ns
    rows.append(dict(live_well=lw, mer_well=mw, n_live=len(A), n_mer=len(B),
                     z=pz, sigma=sig, dx=dx+pre[0], dy=dy+pre[1], angle=th))
    print(f"{lw:>5} {mw:>5} {len(A):>7} {len(B):>7} {pz:>8.2f} {sig:>+7.1f}  "
          f"({dx+pre[0]:>+9.0f},{dy+pre[1]:>+9.0f})")

if not rows:
    raise SystemExit("\nno well pairs had enough cells")
R = pd.DataFrame(rows); R.to_csv(f"{a.out}/well_pairs.csv", index=False)

print("\n" + "="*70); print("DOES THE MAPPING HOLD?"); print("="*70)
good = R[R.sigma > 3]
print(f"  {len(good)}/{len(R)} well pairs score above 3 sigma")
if len(good) >= 2:
    print(f"  offsets among those: dx spread {good.dx.std():.0f} um, "
          f"dy spread {good.dy.std():.0f} um")
    if good.dx.std() < 100 and good.dy.std() < 100:
        print("\n  Those pairs agree on a common offset. That is a real")
        print("  mapping -- use it and match within each well pair.")
    else:
        print("\n  The pairs score well individually but disagree on the")
        print("  offset, so they cannot all be describing the same transform.")
else:
    print("\n  No well pair shows a cell-level alignment under this mapping.")
    print("  The wells may still correspond as wells -- this is about")
    print("  whether the cells inside them do.")
print(f"\n  -> {a.out}/well_pairs.csv")

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def wpanel(ax, P, w, title, colour, labels=True):
    ax.scatter(P[:,0], P[:,1], s=1.2, c=colour, linewidths=0)
    if labels:
        for k in np.unique(w):
            c = P[w==k].mean(0)
            ax.annotate(str(int(k)), c, fontsize=13, weight="bold",
                        ha="center", va="center", color="#111",
                        bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                                  ec="#555", lw=.8, alpha=.85))
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(title, fontsize=12.5); ax.set_xlabel("um")

# every orientation of MERSCOPE, with the live data as reference
mx0, mx1 = M[:,0].min(), M[:,0].max()
my0, my1 = M[:,1].min(), M[:,1].max()
VIEWS = [
    ("MERSCOPE as acquired",      M.copy()),
    ("MERSCOPE x-mirrored",       np.c_[mx0+mx1-M[:,0], M[:,1]]),
    ("MERSCOPE y-mirrored",       np.c_[M[:,0], my0+my1-M[:,1]]),
    ("MERSCOPE rotated 180",      np.c_[mx0+mx1-M[:,0], my0+my1-M[:,1]]),
]
fig, ax = plt.subplots(1, 5, figsize=(26, 7))
wpanel(ax[0], L, wl, f"TIME-LAPSE (reference)\n{len(L):,} cells", "#1D9E75")
for k,(name, V) in enumerate(VIEWS, start=1):
    wpanel(ax[k], V, wells(V, a.ncols, a.nbands), f"{name}\n{len(V):,} cells",
           "#B03A8B")
fig.suptitle("Time-lapse beside MERSCOPE under each orientation "
             "— numbers are well IDs, left to right then top to bottom",
             fontsize=15, y=0.97)
fig.tight_layout(rect=[0,0,1,0.93])
f = f"{a.out}/wells_orientations.png"
fig.savefig(f, dpi=140); plt.close(fig)
print(f"  -> {f}")

# per-pair result, drawn on the layout
fig, ax = plt.subplots(1, 2, figsize=(19, 8))
wpanel(ax[0], L, wl, "live wells", "#1D9E75")
wpanel(ax[1], M, wm, "MERSCOPE wells", "#B03A8B")
ok = {int(r.live_well): r.sigma for _, r in R.iterrows()}
for k in np.unique(wl):
    s_ = ok.get(int(k))
    if s_ is None: continue
    c = L[wl==k].mean(0)
    col = "#1D9E75" if s_ > 3 else "#C0392B"
    ax[0].annotate(f"\u2192{mapping[int(k)]}\n{s_:+.1f}\u03c3", (c[0], c[1]+900),
                   fontsize=9, ha="center", color=col, weight="bold")
fig.suptitle(f"Mapping '{a.mapping}' — sigma above a scrambled null for each "
             f"well pair (green = above 3)", fontsize=14)
fig.tight_layout(rect=[0,0,1,0.94])
f = f"{a.out}/wells_mapping_result.png"
fig.savefig(f, dpi=140); plt.close(fig)
print(f"  -> {f}")
