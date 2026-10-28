#!/usr/bin/env python
"""
Take an offset from identified cell pairs and apply it to everything.

Rather than fitting a transform, this accepts one supplied by hand -- from
cells somebody has identified as corresponding -- and uses it as given.
It then reports what that offset does to the whole dataset: how many live
cells land on a MERSCOPE cell, how that compares with chance, and the
resulting match table.

Two ways in:

  --pairs pairs.csv      a file with live_x,live_y,mer_x,mer_y per row
                         (one row per identified pair). The offset is the
                         mean, and the spread between pairs is reported --
                         that spread is the thing to look at, because
                         pairs that disagree cannot all be correct.

  --offset DX DY         the offset directly, if it is already known.

    python apply_landmark_offset.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope ./flipped/merscope_flipx.csv \
        --pairs dirks_pairs.csv --out ./dirk_offset
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--pairs", default=None)
ap.add_argument("--offset", nargs=2, type=float, default=None)
ap.add_argument("--flip", default="none", help="none|x|y|xy applied to live first")
ap.add_argument("--match-um", type=float, default=7.0,
                help="how close counts as a match")
ap.add_argument("--out", default="./landmark_offset")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
L = live[[lx, ly]].to_numpy(float)
M = mer[["center_x", "center_y"]].to_numpy(float)
sx = -1 if "x" in a.flip else 1
sy = -1 if "y" in a.flip else 1
L = L * np.array([sx, sy], float)

# ------------------------------------------------------------- the offset
if a.pairs:
    P = pd.read_csv(a.pairs)
    need = {"live_x","live_y","mer_x","mer_y"}
    if not need <= set(P.columns):
        raise SystemExit(f"--pairs needs columns {sorted(need)}; "
                         f"found {list(P.columns)}")
    P["dx"] = P.mer_x - P.live_x * sx
    P["dy"] = P.mer_y - P.live_y * sy
    print("PAIRS SUPPLIED"); print("="*58)
    print(P[["live_x","live_y","mer_x","mer_y","dx","dy"]].round(1).to_string(index=False))
    off = np.array([P.dx.mean(), P.dy.mean()])
    spread = np.hypot(P.dx.std(), P.dy.std()) if len(P) > 1 else 0.0
    print(f"\n  mean offset  ({off[0]:+.1f}, {off[1]:+.1f}) um")
    if len(P) > 1:
        print(f"  spread       dx {P.dx.std():.1f}, dy {P.dy.std():.1f} um")
        print(f"\n  The spread is the key number. Pairs that genuinely")
        print(f"  correspond share one offset, so a spread much larger than")
        print(f"  the cell spacing means they cannot all be right.")
else:
    if not a.offset:
        raise SystemExit("give --pairs or --offset")
    off = np.array(a.offset, float); spread = 0.0
    print(f"using offset ({off[0]:+.1f}, {off[1]:+.1f}) um as given")

# ------------------------------------------------------------- apply it
W = L + off
tree = cKDTree(M)
d, j = tree.query(W, k=1)
hit = d < a.match_um

print("\n" + "="*58); print("APPLIED TO EVERY CELL"); print("="*58)
print(f"  live cells                       {len(W):,}")
print(f"  MERSCOPE cells                   {len(M):,}")
print(f"  within {a.match_um:.0f} um of a MERSCOPE cell   "
      f"{hit.sum():,}  ({hit.mean():.1%})")
print(f"  median distance to nearest       {np.median(d):.1f} um")

# what would chance give? scatter the same cells over the same area
rng = np.random.default_rng(0)
lo, hi = M.min(0), M.max(0)
chance = []
for _ in range(5):
    R = rng.uniform(lo, hi, W.shape)
    chance.append((tree.query(R, k=1)[0] < a.match_um).mean())
ch = float(np.mean(chance))
print(f"\n  the same count from random positions   {ch:.1%}")
print(f"  enrichment over chance                 {hit.mean()/max(ch,1e-9):.2f}x")
print("\n  A real alignment gives many times chance. Around 1x means the")
print("  cells landing near a partner are doing so because the field is")
print("  crowded, not because they are the same cells.")

out = pd.DataFrame({
    "live_idx": np.arange(len(W))[hit],
    "fixed_idx": j[hit],
    "dist_um": d[hit],
    "live_x": W[hit,0], "live_y": W[hit,1],
    "mer_x": M[j[hit],0], "mer_y": M[j[hit],1],
})
for c in ("TrackNumber","Position","Colony","Area"):
    if c in live: out[c] = live[c].to_numpy()[out.live_idx]
if "EntityID" in mer: out["fixed_id"] = mer["EntityID"].to_numpy()[out.fixed_idx]
f = f"{a.out}/matches_from_offset.csv"
out.to_csv(f, index=False)
print(f"\n  -> {f}  ({len(out):,} rows)")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(22, 7))
ax[0].scatter(M[:,0], M[:,1], s=.8, c="#E0C4DB", linewidths=0, label="MERSCOPE")
ax[0].scatter(W[:,0], W[:,1], s=.8, c="#1D9E75", linewidths=0, alpha=.6, label="live + offset")
ax[0].scatter(W[hit,0], W[hit,1], s=4, c="#C0392B", label=f"within {a.match_um:.0f} um")
ax[0].set_aspect("equal"); ax[0].invert_yaxis(); ax[0].legend(markerscale=5, fontsize=9)
ax[0].set_title("Everything, with the offset applied", fontsize=12.5)
ax[1].hist(d[d<60], bins=60, color="#357")
ax[1].axvline(a.match_um, color="#C0392B", ls="--", label=f"{a.match_um:.0f} um")
ax[1].set_xlabel("distance to nearest MERSCOPE cell (um)"); ax[1].legend()
ax[1].set_title("A correct offset peaks near zero", fontsize=12.5)
ax[2].bar(["this offset","random"], [hit.mean()*100, ch*100],
          color=["#1D9E75","#9AA7B2"])
ax[2].set_ylabel(f"% of cells within {a.match_um:.0f} um")
ax[2].set_title("Compared with chance", fontsize=12.5)
for i,v in enumerate([hit.mean()*100, ch*100]):
    ax[2].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(f"{a.out}/offset_applied.png", dpi=145)
print(f"  -> {a.out}/offset_applied.png")
