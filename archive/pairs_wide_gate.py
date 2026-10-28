#!/usr/bin/env python
"""
Pair table at a chosen search radius, with the expected error rate attached.

Widening the radius always produces more pairs. It does not produce more
CORRECT pairs: at radius r, with cells spaced s apart, roughly (r/s)^2
MERSCOPE cells sit within reach of any live cell, so picking the nearest
is right about 1 time in (r/s)^2 unless a real alignment is present.

So every row carries an `estimated_correct` figure, the summary states the
expected error rate, and the filename records the radius. A file like this
is easy to mistake for a verified result months later, and the numbers
should travel with it.

Enrichment is the number that says whether the pairs mean anything:
observed matches divided by what the same cells would give scattered at
random over the same area. Near 1.0 means the pairs are proximity, not
correspondence.

    python pairs_wide_gate.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --gate-um 50 --out ./pairs_wide
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--gate-um", type=float, default=50.0)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--flip", default="xy", help="global orientation of the live data")
ap.add_argument("--mirror", default="none", choices=["none","x","y","xy"],
                help="mirror applied to each region block about its own centre")
ap.add_argument("--reorder", default="none", choices=["none","x","y"],
                help="reverse the order of the fields within each region, "
                     "moving each field bodily to the reversed slot while "
                     "its contents stay as they are")
ap.add_argument("--sweep", action="store_true",
                help="report the rate at several radii instead of writing a table")
ap.add_argument("--out", default="./pairs_wide")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
print(f"live orientation '{a.flip}'"
      + (f", each block mirrored '{a.mirror}'" if a.mirror != "none" else "")
      + (f", fields reordered on {a.reorder}" if a.reorder != "none" else "")
      + "\n")
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

def region_data(reg, ps):
    s = live[live.Position.isin(ps)].reset_index()
    g = mer[mer.region == reg].reset_index()
    if len(s) < 10 or len(g) < 10:
        return None
    A = s[[lx, ly]].to_numpy(float) * [sx, sy]
    B = g[["center_x", "center_y"]].to_numpy(float)

    if a.mirror != "none":
        if "x" in a.mirror:
            c = (A[:, 0].min() + A[:, 0].max()) / 2; A[:, 0] = 2*c - A[:, 0]
        if "y" in a.mirror:
            c = (A[:, 1].min() + A[:, 1].max()) / 2; A[:, 1] = 2*c - A[:, 1]

    if a.reorder != "none":
        ai = 0 if a.reorder == "x" else 1
        pos = s.Position.to_numpy()
        cen = {p: A[pos == p].mean(0) for p in np.unique(pos)}
        order = sorted(cen, key=lambda p: cen[p][ai])
        for src, dst in zip(order, order[::-1]):
            A[pos == src] += cen[dst] - cen[src]

    A = A + (B.mean(0) - A.mean(0))
    return s, g, A, B

if a.sweep:
    print(f"{'radius':>8} {'matched':>9} {'by chance':>11} {'enrichment':>11} "
          f"{'est. correct':>13}")
    print("-" * 58)
    for r in (5, 10, 15, 20, 30, 50, 80, 120):
        tot = hit = 0; ch_tot = 0.0
        for reg, ps in groups.items():
            d_ = region_data(reg, ps)
            if d_ is None: continue
            _, _, A, B = d_
            t = cKDTree(B)
            hit += (t.query(A, k=1)[0] < r).sum(); tot += len(A)
            lo_, hi_ = B.min(0), B.max(0)
            ch_tot += np.mean([(t.query(rng.uniform(lo_, hi_, A.shape), k=1)[0]
                                < r).mean() for _ in range(3)]) * len(A)
        real = hit / tot; ch = ch_tot / tot
        enr = real / max(ch, 1e-12)
        corr = max(0.0, (real - ch) / max(real, 1e-12))
        print(f"{r:>7}um {real:>9.1%} {ch:>11.1%} {enr:>10.2f}x {corr:>12.0%}")
    print("\n  'est. correct' is the share of pairs beyond what chance gives.")
    print("  Near 0% means essentially all the pairs are proximity.")
    raise SystemExit(0)

out, summary = [], []
for reg, ps in groups.items():
    d_ = region_data(reg, ps)
    if d_ is None: continue
    s, g, A, B = d_
    t = cKDTree(B)
    d, j = t.query(A, k=1)
    keep = d < a.gate_um
    lo_, hi_ = B.min(0), B.max(0)
    ch = np.mean([(t.query(rng.uniform(lo_, hi_, A.shape), k=1)[0]
                   < a.gate_um).mean() for _ in range(5)])
    real = keep.mean()
    enr = real / max(ch, 1e-12)
    corr = max(0.0, (real - ch) / max(real, 1e-12))
    summary.append(dict(region=reg, n_live=len(A), matched=int(keep.sum()),
                        rate=real, chance=ch, enrichment=enr,
                        est_correct=corr, median_um=float(np.median(d))))
    print(f"{reg:>5}: {keep.sum():5d}/{len(A):5d} ({real:5.1%})   "
          f"chance {ch:5.1%}   enrichment {enr:4.2f}x   "
          f"est. correct {corr:4.0%}")
    out.append(pd.DataFrame({
        "region": reg,
        "Position": s.Position.values[keep],
        "live_row": s["index"].values[keep],
        "live_track": (s.TrackNumber.values[keep] if "TrackNumber" in s
                       else np.nan),
        "fixed_id": (g.EntityID.values[j[keep]] if "EntityID" in g
                     else j[keep]),
        "dist_um": d[keep],
        "estimated_correct": corr,
        "live_x": A[keep, 0], "live_y": A[keep, 1],
        "mer_x": B[j[keep], 0], "mer_y": B[j[keep], 1]}))

T = pd.concat(out, ignore_index=True)
S = pd.DataFrame(summary)
f = f"{a.out}/pairs_gate{a.gate_um:.0f}um.csv"
hdr = (f"# radius {a.gate_um:.0f} um; nearest MERSCOPE cell within that "
       f"radius after block alignment.\n"
       f"# NOT verified correspondences. Overall enrichment over chance "
       f"{S.rate.sum()/max(S.chance.sum(),1e-12):.2f}x; "
       f"estimated share of pairs beyond chance "
       f"{max(0,(S.rate.mean()-S.chance.mean())/max(S.rate.mean(),1e-12)):.0%}.\n")
with open(f, "w") as fh:
    fh.write(hdr); T.to_csv(fh, index=False)
S.to_csv(f"{a.out}/summary_gate{a.gate_um:.0f}um.csv", index=False)

print(f"\n  {len(T):,} pairs from {len(live):,} live cells "
      f"({len(T)/len(live):.1%}) at a {a.gate_um:.0f} um radius")
print(f"  overall enrichment over chance: "
      f"{S.rate.mean()/max(S.chance.mean(),1e-12):.2f}x")
print(f"\n  -> {f}")
print(f"  -> {a.out}/summary_gate{a.gate_um:.0f}um.csv")

# ---------------------------------------------------------------- figures
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

allA, allB, allP = [], [], []
for reg, ps in groups.items():
    d_ = region_data(reg, ps)
    if d_ is None: continue
    _, _, A, B = d_
    allA.append(A); allB.append(B)
    sub = T[T.region == reg]
    allP.append(sub)
A = np.vstack(allA); B = np.vstack(allB); P = pd.concat(allP)

fig, ax = plt.subplots(1, 3, figsize=(22, 8))

# 1. everything, with matched cells picked out
ax[0].scatter(B[:,0], B[:,1], s=1, c="#E3CFDE", linewidths=0, label="MERSCOPE")
ax[0].scatter(A[:,0], A[:,1], s=1, c="#C3E3D4", linewidths=0, label="live")
for _, r in P.iterrows():
    ax[0].plot([r.live_x, r.mer_x], [r.live_y, r.mer_y], "-", c="#444",
               lw=.35, zorder=3)
ax[0].scatter(P.live_x, P.live_y, s=5, c="#1D9E75", zorder=4, label="matched live")
ax[0].scatter(P.mer_x, P.mer_y, s=5, c="#B03A8B", zorder=4, label="matched MERSCOPE")
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].legend(markerscale=4, fontsize=8)
ax[0].set_title(f"{len(P):,} pairs at a {a.gate_um:.0f} um radius", fontsize=12)
ax[0].set_xlabel("um"); ax[0].set_ylabel("um")

# 2. the offsets: a real alignment is one tight cluster
dx = P.mer_x - P.live_x; dy = P.mer_y - P.live_y
ax[1].scatter(dx, dy, s=5, c="#1D6F8C", alpha=.45, linewidths=0)
ax[1].axhline(0, c="#999", lw=.6); ax[1].axvline(0, c="#999", lw=.6)
ax[1].set_aspect("equal")
ax[1].set_title(f"offset of every pair\nspread {np.hypot(dx.std(), dy.std()):.0f} um",
                fontsize=12)
ax[1].set_xlabel("dx (um)"); ax[1].set_ylabel("dy (um)")

# 3. rate against chance, per region
w = 0.38; xs = np.arange(len(S))
ax[2].bar(xs-w/2, S.rate*100, w, color="#1D9E75", label="matched")
ax[2].bar(xs+w/2, S.chance*100, w, color="#9AA7B2", label="chance")
ax[2].set_xticks(xs); ax[2].set_xticklabels(S.region, rotation=45)
ax[2].set_ylabel(f"% within {a.gate_um:.0f} um"); ax[2].legend(fontsize=9)
ax[2].set_title("matched versus chance, by region", fontsize=12)
for i, r in enumerate(S.itertuples()):
    ax[2].text(i, max(r.rate, r.chance)*100, f"{r.enrichment:.2f}x",
               ha="center", va="bottom", fontsize=8)

fig.suptitle(f"Pairs at {a.gate_um:.0f} um — overall enrichment "
             f"{S.rate.mean()/max(S.chance.mean(),1e-12):.2f}x over chance",
             fontsize=14)
fig.tight_layout(rect=[0,0,1,0.94])
g = f"{a.out}/pairs_gate{a.gate_um:.0f}um.png"
fig.savefig(g, dpi=145); plt.close(fig)
print(f"  -> {g}")
print("\n  Middle panel is the one to read: a real alignment puts every")
print("  pair at nearly the same offset, so it would be one tight cluster.")
