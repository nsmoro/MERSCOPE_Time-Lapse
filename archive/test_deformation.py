#!/usr/bin/env python
"""
Can a smooth local deformation explain the mismatch?

Every transform tested so far has been rigid: one shift, rotation, scale
or mirror applied to everything. If cells moved in a locally coherent way
-- a gel swelling unevenly, neighbouring cells carried together while
distant ones went elsewhere -- no rigid transform can recover it.

A flexible warp can fit almost anything, so fitting one proves nothing on
its own. The test here is whether it GENERALISES:

  * fit the deformation using half the cells
  * measure how well it predicts the OTHER half, which it never saw

A real deformation describes the held-out cells about as well as the ones
it was fitted on. A warp that is merely bending to noise fits the training
half and fails the rest. The gap between the two is the answer.

A shuffled control is included: the same fit against deliberately
scrambled correspondences, which shows what "fitting noise" scores.

    python test_deformation.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --positions 22,23,24 --region R18 --out ./deform_R18

    python test_deformation.py ... --all --out ./deform_all
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--positions", default=None)
ap.add_argument("--region", default=None)
ap.add_argument("--all", action="store_true")
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy")
ap.add_argument("--gate-um", type=float, default=60.0,
                help="pairs further apart than this are not used for fitting")
ap.add_argument("--smooth-um", type=float, default=400.0,
                help="how smooth the deformation is allowed to be; smaller "
                     "means more flexible and more able to fit noise")
ap.add_argument("--n-repeats", type=int, default=5)
ap.add_argument("--out", default="./deform")
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
pairs = ([(r, p) for r, p in groups.items()] if a.all else
         [(a.region, [int(p) for p in a.positions.split(",")])])


def rbf_warp(src, dst, query, smooth):
    """Smooth displacement field from src->dst, evaluated at `query`.

    Gaussian-weighted local average of the training displacements. The
    width `smooth` controls how much the field can vary: large means
    nearly rigid, small means it can follow noise.
    """
    if len(src) < 3:
        return query
    d2 = ((query[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * smooth ** 2))
    s = w.sum(1, keepdims=True)
    w = np.divide(w, s, out=np.zeros_like(w), where=s > 1e-12)
    return query + w @ (dst - src)


rows = []
rng = np.random.default_rng(0)
for reg, ps in pairs:
    A = live[live.Position.isin(ps)][[lx, ly]].to_numpy(float) * [sx, sy]
    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    if len(A) < 100 or len(B) < 100:
        print(f"{reg}: too few cells"); continue
    A = A - (A.min(0) + A.max(0)) / 2
    B = B - (B.min(0) + B.max(0)) / 2
    tree = cKDTree(B)

    d0, j0 = tree.query(A, k=1)
    use = d0 < a.gate_um
    if use.sum() < 50:
        print(f"{reg}: only {use.sum()} cells within the gate"); continue

    tr_in, te_in, tr_sh, te_sh = [], [], [], []
    for rep in range(a.n_repeats):
        idx = np.flatnonzero(use)
        rng.shuffle(idx)
        half = len(idx) // 2
        tr, te = idx[:half], idx[half:]

        # real correspondences
        W = rbf_warp(A[tr], B[j0[tr]], A, a.smooth_um)
        dtr, _ = tree.query(W[tr], k=1)
        dte, _ = tree.query(W[te], k=1)
        tr_in.append(np.median(dtr)); te_in.append(np.median(dte))

        # shuffled control: same maths, deliberately wrong targets
        perm = rng.permutation(j0[tr])
        Ws = rbf_warp(A[tr], B[perm], A, a.smooth_um)
        dtrs, _ = tree.query(Ws[tr], k=1)
        dtes, _ = tree.query(Ws[te], k=1)
        tr_sh.append(np.median(dtrs)); te_sh.append(np.median(dtes))

    r = dict(region=reg, n_live=len(A), n_used=int(use.sum()),
             before=float(np.median(d0)),
             fit=float(np.mean(tr_in)), heldout=float(np.mean(te_in)),
             fit_shuf=float(np.mean(tr_sh)), heldout_shuf=float(np.mean(te_sh)))
    rows.append(r)
    print(f"{reg}  ({len(A)} live, {use.sum()} within gate)")
    print(f"    before any warp              {r['before']:6.1f} um")
    print(f"    fitted half                  {r['fit']:6.1f} um")
    print(f"    HELD-OUT half                {r['heldout']:6.1f} um")
    print(f"    held-out, shuffled control   {r['heldout_shuf']:6.1f} um")
    gain = r["before"] - r["heldout"]
    noise = r["before"] - r["heldout_shuf"]
    print(f"    -> real gain {gain:+.1f} um, shuffled gain {noise:+.1f} um\n")

if not rows:
    raise SystemExit("nothing to report")
S = pd.DataFrame(rows); S.to_csv(f"{a.out}/deformation.csv", index=False)

print("=" * 66); print("DOES A DEFORMATION EXPLAIN IT?"); print("=" * 66)
real = (S.before - S.heldout).median()
shuf = (S.before - S.heldout_shuf).median()
print(f"  median improvement on held-out cells, real pairs     {real:+.1f} um")
print(f"  median improvement on held-out cells, shuffled       {shuf:+.1f} um")
print(f"  gap between the fitted and held-out halves           "
      f"{(S.heldout - S.fit).median():+.1f} um")
print()
if real > 2 * max(shuf, 0.5) and real > 10:
    print("  The warp predicts cells it never saw, far better than the")
    print("  shuffled control. That is a real deformation -- worth")
    print("  recovering the matching through it.")
elif real > max(shuf, 0.5) * 1.3:
    print("  A modest real effect above the shuffled control. Worth")
    print("  trying a smaller --smooth-um, but treat it cautiously.")
else:
    print("  The warp does no better on held-out cells than the shuffled")
    print("  control does. It is fitting noise, not deformation -- so a")
    print("  smooth local distortion is not what is missing.")
print(f"\n  -> {a.out}/deformation.csv")
