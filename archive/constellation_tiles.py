#!/usr/bin/env python
"""
Constellation matching tile by tile, instead of a whole region at once.

If different parts of the slide moved differently, a single
constellation match over a whole region would be diluted: the local
neighbourhood patterns would still be shared within a small patch, but
not across the region. Running the test tile by tile looks for that.

Two things to read from the output. Whether any tile scores above its
own control, and -- more telling -- whether the tiles that do score
agree with each other on a transform. One tile scoring could be chance
among many tries; several tiles agreeing could not.

Each tile holds far fewer cells than a whole region, so the power per
tile is lower. The script reports the cell count for each so a weak
result can be told apart from an underpowered one.

    python constellation_tiles.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region R18 --positions 22,23,24 --tiles 3 --out ./const_tiles
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constellation_match import build_quads, vote

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region", required=True)
ap.add_argument("--positions", required=True)
ap.add_argument("--tiles", type=int, default=3)
ap.add_argument("--margin", type=float, default=0.25,
                help="widen the MERSCOPE tile by this fraction, so a cell "
                     "near an edge still has its neighbours")
ap.add_argument("--k-neighbours", type=int, default=8)
ap.add_argument("--max-per-cell", type=int, default=4)
ap.add_argument("--hash-tol", type=float, default=0.05)
ap.add_argument("--scale-min", type=float, default=0.9)
ap.add_argument("--scale-max", type=float, default=1.1)
ap.add_argument("--n-null", type=int, default=3)
ap.add_argument("--min-cells", type=int, default=40)
ap.add_argument("--out", default="./const_tiles")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
ps = [int(p) for p in a.positions.split(",")]
A_all = live[live.Position.isin(ps)][[lx, ly]].to_numpy(float)
B_all = mer[mer.region == a.region][["center_x","center_y"]].to_numpy(float)
print(f"  region {a.region}: {len(A_all):,} live, {len(B_all):,} MERSCOPE")
print(f"  splitting into {a.tiles}x{a.tiles} tiles\n")

def frac(P):
    lo, hi = P.min(0), P.max(0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)

fA, fB = frac(A_all), frac(B_all)
rng = np.random.default_rng(0)
rows = []
print(f"{'tile':>5} {'n_live':>7} {'n_mer':>7} {'quads':>8} {'agree':>6} "
      f"{'null':>10} {'sigma':>7}  {'transform found':>30}")
print("-" * 96)

for ty in range(a.tiles):
    for tx in range(a.tiles):
        t = ty * a.tiles + tx + 1
        lo = np.array([tx, ty]) / a.tiles
        hi = np.array([tx + 1, ty + 1]) / a.tiles
        mA = np.all((fA >= lo) & (fA < hi), axis=1)
        m = a.margin / a.tiles
        mB = np.all((fB >= lo - m) & (fB < hi + m), axis=1)
        A, B = A_all[mA], B_all[mB]
        if len(A) < a.min_cells or len(B) < a.min_cells:
            print(f"{t:>5} {len(A):>7} {len(B):>7}   too few cells")
            continue
        hA, qA = build_quads(A, a.k_neighbours, a.max_per_cell)
        hB, qB = build_quads(B, a.k_neighbours, a.max_per_cell)
        r = vote(A, B, hA, qA, hB, qB, a.hash_tol,
                 (a.scale_min, a.scale_max))
        null = []
        loB, hiB = B.min(0), B.max(0)
        for i in range(a.n_null):
            Bs = rng.uniform(loB, hiB, B.shape)
            hBs, qBs = build_quads(Bs, a.k_neighbours, a.max_per_cell,
                                   seed=i + 1)
            null.append(vote(A, Bs, hA, qA, hBs, qBs, a.hash_tol,
                             (a.scale_min, a.scale_max))["n_consistent"])
        nm, ns = float(np.mean(null)), float(np.std(null) + 1e-9)
        z = (r["n_consistent"] - nm) / ns
        tf = ""
        if r["best"] is not None and r["n_consistent"] > 0:
            s, ang, dx, dy = r["best"]
            tf = f"{s:.3f}x {ang:+.0f}deg ({dx:+.0f},{dy:+.0f})"
        print(f"{t:>5} {len(A):>7} {len(B):>7} {r['n_quad_matches']:>8} "
              f"{r['n_consistent']:>6} {nm:>5.1f}+/-{ns:<4.1f} {z:>+7.1f}  "
              f"{tf:>30}")
        rows.append(dict(tile=t, n_live=len(A), n_mer=len(B),
                         quads=r["n_quad_matches"], agree=r["n_consistent"],
                         null=nm, null_sd=ns, sigma=z,
                         scale=r["best"][0] if r["best"] is not None else np.nan,
                         angle=r["best"][1] if r["best"] is not None else np.nan,
                         dx=r["best"][2] if r["best"] is not None else np.nan,
                         dy=r["best"][3] if r["best"] is not None else np.nan))

if not rows:
    raise SystemExit("\nno tile had enough cells")
S = pd.DataFrame(rows); S.to_csv(f"{a.out}/tiles_{a.region}.csv", index=False)

print("\n" + "=" * 70)
print("DO ANY TILES SHOW A PATTERN?")
print("=" * 70)
hits = S[S.sigma > 3]
print(f"  {len(hits)}/{len(S)} tiles score above 3 sigma")
if len(hits) >= 2:
    print(f"  among those: angle spread {hits.angle.std():.1f} deg, "
          f"shift spread ({hits.dx.std():.0f}, {hits.dy.std():.0f}) um")
    if hits.angle.std() < 5 and np.hypot(hits.dx.std(), hits.dy.std()) < 200:
        print("\n  Several tiles agree on the same transform. That is a real")
        print("  correspondence -- use this transform and match within it.")
    else:
        print("\n  They score individually but disagree on the transform, so")
        print("  they cannot all be describing the same correspondence.")
        print("  With this many tiles tested, a few scoring above 3 sigma")
        print("  is what chance produces.")
elif len(hits) == 1:
    print("\n  One tile out of many scoring could easily be chance. Two or")
    print("  more agreeing on a transform is what would not be.")
else:
    print("\n  No tile shows more agreement than its own scrambled control.")
    print("  The local neighbourhood patterns are not shared, even within")
    print("  small patches.")
print(f"\n  median cells per tile: {S.n_live.median():.0f} live, "
      f"{S.n_mer.median():.0f} MERSCOPE")
if S.n_live.median() < 150:
    print("  (that is on the low side -- the method was validated at 400,")
    print("   so a weak result here is partly a matter of power)")
print(f"\n  -> {a.out}/tiles_{a.region}.csv")
