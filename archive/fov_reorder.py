#!/usr/bin/env python
"""
Mirror a region, then reorder its fields of view without flipping them.

Two steps, and they are different in kind:

  1. a mirror applied to the whole block
  2. the fields swapped along one axis -- the first becomes the last --
     while the cells INSIDE each field keep their arrangement

The second step is the unusual one. Every transform tested so far moves
all the cells together, so reversing the field ORDER while leaving each
field internally unchanged is something a global search cannot express.
It is what you would get if the stage positions were enumerated in the
opposite order from the way they physically sit.

    python fov_reorder.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --positions 22,23,24 --region R18 \
        --flip xy --mirror x --reorder y --out ./reorder_R18

    python fov_reorder.py ... --all --out ./reorder_all
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
ap.add_argument("--flip", default="xy", help="global orientation")
ap.add_argument("--mirror", default="x", choices=["none", "x", "y", "xy"],
                help="mirror applied to the whole block")
ap.add_argument("--reorder", default="y", choices=["none", "x", "y"],
                help="axis along which to reverse the field order")
ap.add_argument("--point-size", type=float, default=6.0)
ap.add_argument("--out", default="./reorder")
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


def reorder_fovs(df, xcol, ycol, axis):
    """Reverse the order of the fields along `axis`, moving each field's
    cells to the slot the reversed order gives it. The arrangement inside
    each field is untouched -- only which slot it occupies changes."""
    out = df.copy()
    ai = xcol if axis == "x" else ycol
    cen = df.groupby("Position")[[xcol, ycol]].mean()
    order = cen[ai].sort_values().index.tolist()      # as laid out
    target = order[::-1]                              # reversed
    for src, dst in zip(order, target):
        shift = cen.loc[dst] - cen.loc[src]
        m = out.Position == src
        out.loc[m, xcol] = out.loc[m, xcol] + shift[xcol]
        out.loc[m, ycol] = out.loc[m, ycol] + shift[ycol]
    return out


import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = []

for reg, ps in pairs:
    sub = live[live.Position.isin(ps)].copy()
    if len(sub) < 10:
        print(f"{reg}: too few cells"); continue
    sub["px"] = sub[lx] * sx
    sub["py"] = sub[ly] * sy

    stages = {"as is": sub.copy()}

    m = sub.copy()
    if "x" in a.mirror:
        c = (m.px.min() + m.px.max()) / 2
        m["px"] = 2 * c - m.px
    if "y" in a.mirror:
        c = (m.py.min() + m.py.max()) / 2
        m["py"] = 2 * c - m.py
    stages[f"mirrored {a.mirror}"] = m

    r = reorder_fovs(m, "px", "py", a.reorder) if a.reorder != "none" else m
    stages[f"+ fields reordered on {a.reorder}"] = r

    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    B = B - (B.min(0) + B.max(0)) / 2
    tree = cKDTree(B)

    res = {}
    for name, df in stages.items():
        P = df[["px", "py"]].to_numpy(float)
        P = P - (P.min(0) + P.max(0)) / 2
        d, _ = tree.query(P, k=1)
        res[name] = (P, float(np.median(d)), float((d < 10).mean()))

    best = min(res, key=lambda k: res[k][1])
    print(f"{reg}  ({len(sub)} live, {len(B)} MERSCOPE)")
    for name, (_, med, pct) in res.items():
        mark = "  <-- best" if name == best else ""
        print(f"    {name:<34} median {med:6.1f} um   "
              f"{pct:5.1%} within 10 um{mark}")
    rows.append(dict(region=reg, **{f"median_{k}": v[1] for k, v in res.items()},
                     best=best))

    lim = max(max(np.abs(P).max() for P, _, _ in res.values()),
              np.abs(B).max()) * 1.06
    fig, ax = plt.subplots(1, len(res) + 1, figsize=(5.3 * (len(res) + 1), 7))
    for k, (name, (P, med, pct)) in enumerate(res.items()):
        ax[k].scatter(B[:, 0], B[:, 1], s=a.point_size, c="#B03A8B",
                      linewidths=0, alpha=.45)
        ax[k].scatter(P[:, 0], P[:, 1], s=a.point_size, c="#1D9E75",
                      linewidths=0, alpha=.6)
        ax[k].set_title(f"{name}\nmedian {med:.0f} um", fontsize=11,
                        color="#1A7A4C" if name == best else "black")
    P = res[best][0]
    ax[-1].scatter(B[:, 0], B[:, 1], s=a.point_size * 1.4, c="#B03A8B",
                   linewidths=0, alpha=.5, label="MERSCOPE")
    ax[-1].scatter(P[:, 0], P[:, 1], s=a.point_size * 1.4, c="#1D9E75",
                   linewidths=0, alpha=.5, label="live, best")
    ax[-1].set_title("best overlay", fontsize=11); ax[-1].legend(fontsize=9,
                                                                 markerscale=2)
    for A_ in ax:
        A_.set_aspect("equal"); A_.set_xlim(-lim, lim); A_.set_ylim(lim, -lim)
        A_.set_xlabel("um"); A_.grid(alpha=.18, lw=.5)
    ax[0].set_ylabel("um")
    fig.suptitle(f"{reg} — mirror '{a.mirror}', then reverse the field order "
                 f"along {a.reorder}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f = f"{a.out}/reorder_{reg}.png"
    fig.savefig(f, dpi=145); plt.close(fig)
    print(f"    -> {f}\n")

if rows:
    S = pd.DataFrame(rows); S.to_csv(f"{a.out}/reorder_summary.csv", index=False)
    if len(S) > 1:
        print("=" * 64)
        print(f"  best option per region: {dict(S.best.value_counts())}")
        cols = [c for c in S.columns if c.startswith("median_")]
        print("  median distance, averaged over regions:")
        for c in cols:
            print(f"    {c[7:]:<34} {S[c].median():6.1f} um")
    print(f"\n  -> {a.out}/reorder_summary.csv")
