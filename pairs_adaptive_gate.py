#!/usr/bin/env python
"""
Pair table with a search radius chosen per region from that region's own fit.

Regions align to different accuracy, so a single radius is either too
tight for the worst or too loose for the best. This measures each
region's nearest-neighbour distance distribution and sets that region's
radius from it.

A caution about the resulting match rate. If the radius is set at a
region's MEDIAN nearest-neighbour distance, then by construction about
half that region's cells fall inside it -- whether or not any real
correspondence exists. The percentage then restates the threshold rather
than measuring anything. The enrichment column is what remains
informative: the match rate divided by what the same cells would give
scattered at random over the same area.

    # radius = each region's median nearest-neighbour distance
    python pairs_adaptive_gate.py --live ... --merscope ... --stat median

    # or a percentile, or a multiple of the cell spacing
    python pairs_adaptive_gate.py ... --stat p25
    python pairs_adaptive_gate.py ... --stat spacing --spacing-mult 0.9
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy")
ap.add_argument("--mirror", default="none", choices=["none","x","y","xy"])
ap.add_argument("--reorder", default="none", choices=["none","x","y"])
ap.add_argument("--stat", default="median",
                choices=["median", "p25", "p10", "spacing"],
                help="how each region's radius is set")
ap.add_argument("--spacing-mult", type=float, default=0.9,
                help="with --stat spacing: radius = this x cell spacing")
ap.add_argument("--pad-um", type=float, default=0.0,
                help="add this many microns to each region's radius after "
                     "the statistic is computed")
ap.add_argument("--out", default="./pairs_adaptive")
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

rng = np.random.default_rng(0)


def prepare(reg, ps):
    s = live[live.Position.isin(ps)].reset_index()
    g = mer[mer.region == reg].reset_index()
    if len(s) < 10 or len(g) < 10:
        return None
    A = s[[lx, ly]].to_numpy(float) * [sx, sy]
    B = g[["center_x", "center_y"]].to_numpy(float)
    if a.mirror != "none":
        if "x" in a.mirror:
            c = (A[:,0].min() + A[:,0].max())/2; A[:,0] = 2*c - A[:,0]
        if "y" in a.mirror:
            c = (A[:,1].min() + A[:,1].max())/2; A[:,1] = 2*c - A[:,1]
    if a.reorder != "none":
        ai = 0 if a.reorder == "x" else 1
        pos = s.Position.to_numpy()
        cen = {p: A[pos == p].mean(0) for p in np.unique(pos)}
        order = sorted(cen, key=lambda p: cen[p][ai])
        for src, dst in zip(order, order[::-1]):
            A[pos == src] += cen[dst] - cen[src]
    A = A + (B.mean(0) - A.mean(0))
    return s, g, A, B


if a.pad_um:
    print(f"radius = each region's {a.stat} + {a.pad_um:.0f} um\n")
print(f"{'region':>7} {'n_live':>7} {'n_mer':>7} {'spacing':>9} "
      f"{'radius':>8} {'x spacing':>10} {'matched':>9} {'chance':>8} "
      f"{'enrich':>7}")
print("-" * 70)
out, summary = [], []
for reg in regions:
    d_ = prepare(reg, groups[reg])
    if d_ is None:
        continue
    s, g, A, B = d_
    tree = cKDTree(B)
    d, j = tree.query(A, k=1)
    spacing = float(np.median(cKDTree(B).query(B, k=2)[0][:, 1]))

    if a.stat == "median":
        gate = float(np.median(d))
    elif a.stat == "p25":
        gate = float(np.percentile(d, 25))
    elif a.stat == "p10":
        gate = float(np.percentile(d, 10))
    else:
        gate = a.spacing_mult * spacing
    base = gate
    gate = gate + a.pad_um

    keep = d < gate
    lo_, hi_ = B.min(0), B.max(0)
    ch = float(np.mean([(tree.query(rng.uniform(lo_, hi_, A.shape), k=1)[0]
                         < gate).mean() for _ in range(5)]))
    rate = float(keep.mean())
    print(f"{reg:>7} {len(A):>7} {len(B):>7} {spacing:>8.1f}u "
          f"{gate:>7.1f}u {gate/spacing:>9.1f}x {rate:>8.1%} {ch:>7.1%} "
          f"{rate/max(ch,1e-12):>6.2f}x")
    summary.append(dict(region=reg, n_live=len(A), n_mer=len(B),
                        spacing_um=spacing, base_um=base, radius_um=gate,
                        matched=int(keep.sum()), rate=rate, chance=ch,
                        enrichment=rate/max(ch,1e-12),
                        median_um=float(np.median(d))))
    out.append(pd.DataFrame({
        "region": reg, "radius_um": gate,
        "Position": s.Position.values[keep],
        "live_row": s["index"].values[keep],
        "live_track": (s.TrackNumber.values[keep] if "TrackNumber" in s
                       else np.nan),
        "fixed_id": (g.EntityID.values[j[keep]] if "EntityID" in g
                     else j[keep]),
        "dist_um": d[keep],
        "live_x": A[keep,0], "live_y": A[keep,1],
        "mer_x": B[j[keep],0], "mer_y": B[j[keep],1]}))

T = pd.concat(out, ignore_index=True)
S = pd.DataFrame(summary)
tag = a.stat + (f"_plus{a.pad_um:.0f}" if a.pad_um else "")
f = f"{a.out}/pairs_adaptive_{tag}.csv"
hdr = (f"# radius set per region from its own nearest-neighbour "
       f"distribution ({a.stat}).\n"
       f"# NOT verified correspondences. Overall enrichment over chance "
       f"{S.rate.mean()/max(S.chance.mean(),1e-12):.2f}x.\n")
with open(f, "w") as fh:
    fh.write(hdr); T.to_csv(fh, index=False)
S.to_csv(f"{a.out}/summary_{tag}.csv", index=False)

print("\n" + "=" * 70)
print(f"  {len(T):,} pairs from {len(live):,} live cells "
      f"({len(T)/len(live):.1%})")
print(f"  overall enrichment over chance: "
      f"{S.rate.mean()/max(S.chance.mean(),1e-12):.2f}x")
print(f"  radii used: {S.radius_um.min():.0f} to {S.radius_um.max():.0f} um")
print(f"  cell spacing: {S.spacing_um.min():.0f} to "
      f"{S.spacing_um.max():.0f} um")
if a.stat == "median" and not a.pad_um:
    print("\n  Note: with the radius AT each region's median distance,")
    print("  about half of each region's cells fall inside it by")
    print("  construction. The match rate restates the threshold; the")
    print("  enrichment column is the part that measures something.")
elif a.pad_um:
    print(f"\n  Each radius is now {S.radius_um.div(S.spacing_um).min():.1f}"
          f" to {S.radius_um.div(S.spacing_um).max():.1f} times the local")
    print("  cell spacing. Beyond about one spacing, a live cell has")
    print("  several MERSCOPE cells within reach, so extra matches are")
    print("  nearer neighbours rather than better ones -- watch whether")
    print("  the enrichment column moves at all as the radius grows.")
print(f"\n  -> {f}")
print(f"  -> {a.out}/summary_{tag}.csv")
