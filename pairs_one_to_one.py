#!/usr/bin/env python
"""
One-to-one pairing: each live cell to at most one MERSCOPE cell, and the
other way round.

Nearest-neighbour matching lets several live cells claim the same
MERSCOPE partner, which cannot be right -- a cell has one counterpart or
none. Two ways to enforce that:

  mutual   keep a pair only when each is the other's nearest. Strict,
           fast, and a good filter: a pair that survives is one where
           neither cell has a better candidate.

  optimal  the assignment that minimises total distance over all pairs
           at once. Finds more pairs than mutual, because it can resolve
           competition sensibly rather than discarding it, but it will
           also pair cells that both had better options taken from them.

Enforcing one-to-one reduces the count. That is the point: the pairs
that remain are the ones where the claim is unambiguous.

    python pairs_one_to_one.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --flip xy --mirror y --reorder y \
        --gate-um 52 --method both --out ./one_to_one
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy")
ap.add_argument("--mirror", default="none", choices=["none","x","y","xy"])
ap.add_argument("--reorder", default="none", choices=["none","x","y"])
ap.add_argument("--gate-um", type=float, default=52.0,
                help="fixed radius; ignored when --stat is given")
ap.add_argument("--stat", default=None,
                choices=["median", "p25", "p10", "spacing"],
                help="set each region's radius from its own fit instead of "
                     "using one fixed value")
ap.add_argument("--pad-um", type=float, default=0.0,
                help="add this to each region's radius")
ap.add_argument("--spacing-mult", type=float, default=0.9)
ap.add_argument("--method", default="both",
                choices=["mutual", "optimal", "both"])
ap.add_argument("--out", default="./one_to_one")
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
            c = (A[:,0].min()+A[:,0].max())/2; A[:,0] = 2*c - A[:,0]
        if "y" in a.mirror:
            c = (A[:,1].min()+A[:,1].max())/2; A[:,1] = 2*c - A[:,1]
    if a.reorder != "none":
        ai = 0 if a.reorder == "x" else 1
        pos = s.Position.to_numpy()
        cen = {q: A[pos == q].mean(0) for q in np.unique(pos)}
        order = sorted(cen, key=lambda q: cen[q][ai])
        for src, dst in zip(order, order[::-1]):
            A[pos == src] += cen[dst] - cen[src]
    A = A + (B.mean(0) - A.mean(0))
    return s, g, A, B


def mutual_pairs(A, B, gate):
    ta, tb = cKDTree(A), cKDTree(B)
    da, ja = tb.query(A, k=1)        # each live cell's nearest MERSCOPE
    db, jb = ta.query(B, k=1)        # each MERSCOPE cell's nearest live
    i = np.arange(len(A))
    keep = (da < gate) & (jb[ja] == i)
    return i[keep], ja[keep], da[keep]


def optimal_pairs(A, B, gate):
    """Least-total-distance one-to-one assignment, within the gate."""
    ta = cKDTree(A); tb = cKDTree(B)
    cand = ta.query_ball_tree(tb, r=gate)
    rows = [i for i, js in enumerate(cand) for _ in js]
    cols = [j for js in cand for j in js]
    if not rows:
        return np.array([], int), np.array([], int), np.array([])
    ui = np.unique(rows); uj = np.unique(cols)
    mi = {v: k for k, v in enumerate(ui)}; mj = {v: k for k, v in enumerate(uj)}
    BIG = gate * 10
    C = np.full((len(ui), len(uj)), BIG)
    d = np.hypot(A[rows, 0] - B[cols, 0], A[rows, 1] - B[cols, 1])
    for r, c, dist in zip(rows, cols, d):
        C[mi[r], mj[c]] = dist
    ri, ci = linear_sum_assignment(C)
    ok = C[ri, ci] < gate
    return ui[ri[ok]], uj[ci[ok]], C[ri[ok], ci[ok]]


methods = (["mutual", "optimal"] if a.method == "both" else [a.method])
for meth in methods:
    lab = (f"radius per region: {a.stat}" + (f" + {a.pad_um:.0f} um"
           if a.pad_um else "")) if a.stat else f"gate {a.gate_um:.0f} um"
    print(f"\n{'='*86}\n{meth.upper()} one-to-one, {lab}\n{'='*86}")
    print(f"{'region':>7} {'n_live':>7} {'n_mer':>7} {'radius':>8} "
          f"{'nearest':>8} {'one-to-one':>10} {'rate':>7} {'chance':>8} "
          f"{'enrich':>7} {'median':>8}")
    print("-" * 80)
    out, summ = [], []
    for reg in regions:
        d_ = prepare(reg, groups[reg])
        if d_ is None:
            continue
        s, g, A, B = d_
        dnn, _ = cKDTree(B).query(A, k=1)

        # the radius for THIS region
        if a.stat is None:
            gate = a.gate_um
        elif a.stat == "median":
            gate = float(np.median(dnn))
        elif a.stat == "p25":
            gate = float(np.percentile(dnn, 25))
        elif a.stat == "p10":
            gate = float(np.percentile(dnn, 10))
        else:
            sp = float(np.median(cKDTree(B).query(B, k=2)[0][:, 1]))
            gate = a.spacing_mult * sp
        gate += a.pad_um

        n_nn = int((dnn < gate).sum())
        ia, jb, dd = (mutual_pairs if meth == "mutual" else optimal_pairs)(
            A, B, gate)
        # chance: the same procedure against scattered cells
        lo_, hi_ = B.min(0), B.max(0)
        ch = []
        for _ in range(3):
            Bs = rng.uniform(lo_, hi_, B.shape)
            r_ = (mutual_pairs if meth == "mutual" else optimal_pairs)(
                A, Bs, gate)
            ch.append(len(r_[0]) / len(A))
        chm = float(np.mean(ch))
        rate = len(ia) / len(A)
        print(f"{reg:>7} {len(A):>7} {len(B):>7} {gate:>7.1f}u {n_nn:>8} "
              f"{len(ia):>10} {rate:>6.1%} {chm:>7.1%} "
              f"{rate/max(chm,1e-12):>6.2f}x "
              f"{(np.median(dd) if len(dd) else np.nan):>7.1f}u")
        summ.append(dict(region=reg, n_live=len(A), n_mer=len(B),
                         radius_um=gate, nearest=n_nn,
                         one_to_one=len(ia), rate=rate,
                         chance=chm, enrichment=rate/max(chm,1e-12),
                         median_um=float(np.median(dd)) if len(dd) else np.nan))
        if len(ia):
            out.append(pd.DataFrame({
                "region": reg, "method": meth, "radius_um": gate,
                "Position": s.Position.values[ia],
                "live_row": s["index"].values[ia],
                "live_track": (s.TrackNumber.values[ia]
                               if "TrackNumber" in s else np.nan),
                "fixed_id": (g.EntityID.values[jb] if "EntityID" in g else jb),
                "dist_um": dd,
                "live_x": A[ia,0], "live_y": A[ia,1],
                "mer_x": B[jb,0], "mer_y": B[jb,1]}))
    if not out:
        print("  nothing paired"); continue
    T = pd.concat(out, ignore_index=True)
    S = pd.DataFrame(summ)
    tag = (f"{a.stat}" + (f"plus{a.pad_um:.0f}" if a.pad_um else "")
           ) if a.stat else f"{a.gate_um:.0f}um"
    f = f"{a.out}/pairs_{meth}_{tag}.csv"
    T.to_csv(f, index=False)
    S.to_csv(f"{a.out}/summary_{meth}_{tag}.csv", index=False)
    # every live cell and every MERSCOPE cell used at most once?
    assert T.groupby("region").live_row.apply(lambda v: v.is_unique).all()
    assert T.groupby("region").fixed_id.apply(lambda v: v.is_unique).all()
    print(f"\n  {len(T):,} pairs from {len(live):,} live cells "
          f"({len(T)/len(live):.1%})")
    print(f"  each cell used at most once on both sides: verified")
    print(f"  overall enrichment over chance: "
          f"{S.rate.mean()/max(S.chance.mean(),1e-12):.2f}x")
    print(f"  -> {f}")
