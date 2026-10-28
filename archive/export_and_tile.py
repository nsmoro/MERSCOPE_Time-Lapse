#!/usr/bin/env python
"""
Write out the two centroid sets in a chosen orientation, tile them, and
look for shared local patterns tile by tile.

Three steps in one place so the inputs are visible rather than implied:

  1. export the live centroids and the MERSCOPE centroids, with the
     chosen rotation applied to whichever side you name, as plain CSVs
  2. cut both into a grid of tiles covering the same part of the slide
  3. in each tile, test whether the neighbourhood patterns are shared,
     against that tile's own scrambled control

Note that rotating one side by 180 degrees gives the same correspondence
as rotating the other by 180 degrees. Which side carries the rotation is
a matter of which frame you want the output in, not a different test.

    python export_and_tile.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --rotate merscope --tiles 4 --out ./tiled
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constellation_match import build_quads, vote

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--rotate", default="merscope",
                choices=["merscope", "live", "none"],
                help="which side gets the 180 degree rotation")
ap.add_argument("--region", default=None,
                help="restrict to one MERSCOPE region and its positions")
ap.add_argument("--positions", default=None)
ap.add_argument("--tiles", type=int, default=4)
ap.add_argument("--margin", type=float, default=0.25)
ap.add_argument("--k-neighbours", type=int, default=8)
ap.add_argument("--hash-tol", type=float, default=0.05)
ap.add_argument("--min-cells", type=int, default=40)
ap.add_argument("--n-null", type=int, default=3)
ap.add_argument("--out", default="./tiled")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
if a.positions:
    live = live[live.Position.isin([int(p) for p in a.positions.split(",")])]
if a.region:
    mer = mer[mer.region == a.region]

A = live[[lx, ly]].to_numpy(float)
B = mer[["center_x", "center_y"]].to_numpy(float)

def rot180(P):
    lo, hi = P.min(0), P.max(0)
    return (lo + hi) - P

if a.rotate == "merscope":
    B = rot180(B)
elif a.rotate == "live":
    A = rot180(A)
print(f"  rotation applied to: {a.rotate}")
print(f"  live     {len(A):,} cells  "
      f"x {A[:,0].min():.0f}..{A[:,0].max():.0f}  "
      f"y {A[:,1].min():.0f}..{A[:,1].max():.0f}")
print(f"  MERSCOPE {len(B):,} cells  "
      f"x {B[:,0].min():.0f}..{B[:,0].max():.0f}  "
      f"y {B[:,1].min():.0f}..{B[:,1].max():.0f}")

# ---- 1. the CSVs -------------------------------------------------------
dl = live.copy(); dl["x_oriented"] = A[:, 0]; dl["y_oriented"] = A[:, 1]
dm = mer.copy(); dm["x_oriented"] = B[:, 0]; dm["y_oriented"] = B[:, 1]
fl = f"{a.out}/live_oriented.csv"; fm = f"{a.out}/merscope_oriented.csv"
dl.to_csv(fl, index=False); dm.to_csv(fm, index=False)
print(f"\n  -> {fl}")
print(f"  -> {fm}")

# ---- 2 and 3. tiles and patterns ---------------------------------------
def frac(P):
    lo, hi = P.min(0), P.max(0)
    return (P - lo) / np.maximum(hi - lo, 1e-9)

fA, fB = frac(A), frac(B)
rng = np.random.default_rng(0)
rows = []
print(f"\n  {a.tiles}x{a.tiles} tiles\n")
print(f"{'tile':>5} {'n_live':>7} {'n_mer':>7} {'quads':>7} {'agree':>6} "
      f"{'null':>11} {'sigma':>7}  transform")
print("-" * 92)
for ty in range(a.tiles):
    for tx in range(a.tiles):
        t = ty * a.tiles + tx + 1
        lo = np.array([tx, ty]) / a.tiles
        hi = np.array([tx + 1, ty + 1]) / a.tiles
        m = a.margin / a.tiles
        sa = np.all((fA >= lo) & (fA < hi), axis=1)
        sb = np.all((fB >= lo - m) & (fB < hi + m), axis=1)
        At, Bt = A[sa], B[sb]
        if len(At) < a.min_cells or len(Bt) < a.min_cells:
            continue
        hA, qA = build_quads(At, a.k_neighbours, 4)
        hB, qB = build_quads(Bt, a.k_neighbours, 4)
        r = vote(At, Bt, hA, qA, hB, qB, a.hash_tol, (0.9, 1.1))
        null = []
        loB, hiB = Bt.min(0), Bt.max(0)
        for i in range(a.n_null):
            Bs = rng.uniform(loB, hiB, Bt.shape)
            hBs, qBs = build_quads(Bs, a.k_neighbours, 4, seed=i + 1)
            null.append(vote(At, Bs, hA, qA, hBs, qBs, a.hash_tol,
                             (0.9, 1.1))["n_consistent"])
        nm = float(np.mean(null)); ns = float(np.std(null))
        # with a null of zero spread, a plain z-score explodes; compare
        # against the count itself instead
        z = ((r["n_consistent"] - nm) / ns) if ns > 0.3 else \
            (np.nan if r["n_consistent"] <= max(nm, 1) + 1 else 99.0)
        tf = ""
        if r["best"] is not None and r["n_consistent"] > 0:
            s_, ang, dx, dy = r["best"]
            tf = f"{s_:.3f}x {ang:+.0f}deg ({dx:+.0f},{dy:+.0f})"
        zs = "  n/a" if np.isnan(z) else f"{z:>+7.1f}"
        print(f"{t:>5} {len(At):>7} {len(Bt):>7} {r['n_quad_matches']:>7} "
              f"{r['n_consistent']:>6} {nm:>5.1f}+/-{ns:<4.1f} {zs}  {tf}")
        rows.append(dict(tile=t, n_live=len(At), n_mer=len(Bt),
                         quads=r["n_quad_matches"], agree=r["n_consistent"],
                         null=nm, null_sd=ns, sigma=z,
                         **(dict(zip(("scale","angle","dx","dy"), r["best"]))
                            if r["best"] is not None else {})))

if not rows:
    raise SystemExit("\n  no tile had enough cells; try fewer tiles")
S = pd.DataFrame(rows); S.to_csv(f"{a.out}/tiles.csv", index=False)

print("\n" + "=" * 70)
print("DO ANY TILES SHARE A PATTERN?")
print("=" * 70)
print(f"  tiles tested: {len(S)}")
print(f"  agreeing constellations per tile: median {S.agree.median():.0f}, "
      f"max {S.agree.max():.0f}")
print(f"  (the method was validated at 20 agreeing on known-matching data)")
hits = S[(S.agree >= 5) & (S.agree > S.null + 3)]
if len(hits) >= 2:
    print(f"\n  {len(hits)} tiles with real agreement; angle spread "
          f"{hits.angle.std():.1f} deg")
    if hits.angle.std() < 5:
        print("  They agree with each other too. That is a correspondence.")
    else:
        print("  But they disagree on the transform, so not all can be right.")
elif len(hits) == 1:
    print("\n  One tile only. Among this many, that is what chance gives.")
else:
    print("\n  No tile reaches even 5 agreeing constellations. The local")
    print("  patterns are not shared at this scale.")
print(f"\n  -> {a.out}/tiles.csv")
