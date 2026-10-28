#!/usr/bin/env python
"""
Fold one region onto the other, the way a book closes.

Two panels side by side are two pages. Closing the book swings the left
page about the spine onto the right one, so a point d to the left of the
spine lands d to the right of it. The left page's OUTER edge meets the
right page's OUTER edge.

That is a mirror, not a rotation, and it is a different transform from
anything a global search covers: the blocks stay where they are and only
the cells inside each one reverse left-to-right.

Draws the book open, mid-fold and closed, and reports how well the two
agree before and after, so the fold can be judged by a number as well as
by eye.

    python book_fold.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --positions 22,23,24 --region R18 --flip xy --out ./fold_R18

    python book_fold.py ... --all --out ./fold_all
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
ap.add_argument("--flip", default="xy", help="global orientation of the live data")
ap.add_argument("--axis", default="x", choices=["x", "y"],
                help="x = book opens left-right, y = opens top-bottom")
ap.add_argument("--point-size", type=float, default=6.0)
ap.add_argument("--out", default="./fold")
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
if not a.all and not (a.region and a.positions):
    raise SystemExit("give --positions and --region, or --all")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ax_i = 0 if a.axis == "x" else 1
summary = []

for reg, ps in pairs:
    A = live[live.Position.isin(ps)][[lx, ly]].to_numpy(float) * [sx, sy]
    B = mer[mer.region == reg][["center_x", "center_y"]].to_numpy(float)
    if len(A) < 10 or len(B) < 10:
        print(f"{reg}: too few cells"); continue

    # centre both, so only shape is compared
    A = A - (A.min(0) + A.max(0)) / 2
    B = B - (B.min(0) + B.max(0)) / 2

    folded = A.copy()
    folded[:, ax_i] = -folded[:, ax_i]        # the fold

    tree = cKDTree(B)
    d_open, _ = tree.query(A, k=1)
    d_fold, _ = tree.query(folded, k=1)
    summary.append(dict(region=reg, n_live=len(A), n_mer=len(B),
                        median_open=np.median(d_open),
                        median_folded=np.median(d_fold),
                        pct10_open=(d_open < 10).mean(),
                        pct10_folded=(d_fold < 10).mean()))
    better = "FOLDED is closer" if np.median(d_fold) < np.median(d_open) \
        else "open is closer"
    print(f"{reg}  live {len(A):5d}  mer {len(B):5d}   "
          f"median open {np.median(d_open):6.1f} um  "
          f"folded {np.median(d_fold):6.1f} um   -> {better}")

    lim = max(np.abs(A).max(), np.abs(B).max()) * 1.08
    gap = lim * 0.35
    fig, ax = plt.subplots(1, 3, figsize=(20, 7.5))

    # 1. open book
    off = lim + gap
    ax[0].scatter(A[:, 0] - off, A[:, 1], s=a.point_size, c="#1D9E75",
                  linewidths=0, label="live (left page)")
    ax[0].scatter(B[:, 0] + off, B[:, 1], s=a.point_size, c="#B03A8B",
                  linewidths=0, label="MERSCOPE (right page)")
    ax[0].axvline(0, color="#333", lw=2.5, ls="--")
    ax[0].annotate("spine", (0, -lim * 0.92), fontsize=11, ha="center",
                   color="#333", weight="bold")
    for s_, lab in [(-off, "outer"), (off, "outer")]:
        ax[0].annotate(lab, (s_ + (-lim*0.8 if s_ < 0 else lim*0.8), lim*0.92),
                       fontsize=9, ha="center", color="#666")
    ax[0].set_title("book open", fontsize=13)
    ax[0].legend(markerscale=2.2, fontsize=9, loc="upper left")
    ax[0].set_xlim(-2 * off, 2 * off)

    # 2. mid-fold
    half = A.copy()
    half[:, ax_i] *= -0.35
    ax[1].scatter(half[:, 0] - off * 0.45, half[:, 1], s=a.point_size,
                  c="#1D9E75", linewidths=0, alpha=.75)
    ax[1].scatter(B[:, 0] + off, B[:, 1], s=a.point_size, c="#B03A8B",
                  linewidths=0)
    ax[1].axvline(0, color="#333", lw=2.5, ls="--")
    ax[1].annotate("", xy=(off * 0.6, 0), xytext=(-off * 0.6, 0),
                   arrowprops=dict(arrowstyle="->", lw=2.5, color="#1D6F8C",
                                   connectionstyle="arc3,rad=-0.45"))
    ax[1].set_title("closing", fontsize=13)
    ax[1].set_xlim(-2 * off, 2 * off)

    # 3. closed
    ax[2].scatter(B[:, 0], B[:, 1], s=a.point_size * 1.3, c="#B03A8B",
                  linewidths=0, alpha=.55, label="MERSCOPE")
    ax[2].scatter(folded[:, 0], folded[:, 1], s=a.point_size * 1.3,
                  c="#1D9E75", linewidths=0, alpha=.55, label="live, folded")
    ax[2].set_title(f"book closed — median {np.median(d_fold):.0f} um "
                    f"(was {np.median(d_open):.0f})", fontsize=13)
    ax[2].legend(markerscale=2.2, fontsize=9)
    ax[2].set_xlim(-lim, lim)

    for A_ in ax:
        A_.set_aspect("equal"); A_.set_ylim(lim, -lim)
        A_.set_xlabel("um (centred)"); A_.grid(alpha=.18, lw=.5)
    ax[0].set_ylabel("um (centred)")
    fig.suptitle(f"{reg}  —  folding the live page onto MERSCOPE "
                 f"about the {a.axis} spine", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    f = f"{a.out}/fold_{reg}.png"
    fig.savefig(f, dpi=150); plt.close(fig)
    print(f"   -> {f}")

S = pd.DataFrame(summary)
S.to_csv(f"{a.out}/fold_summary.csv", index=False)
if len(S) > 1:
    print("\n" + "=" * 62)
    nb = (S.median_folded < S.median_open).sum()
    print(f"  folded is closer in {nb}/{len(S)} regions")
    print(f"  median distance  open {S.median_open.median():.1f} um   "
          f"folded {S.median_folded.median():.1f} um")
    print(f"  within 10 um     open {S.pct10_open.mean():.1%}   "
          f"folded {S.pct10_folded.mean():.1%}")
    if S.median_folded.median() < 0.5 * S.median_open.median():
        print("\n  The fold is a large improvement — this looks like the")
        print("  missing transform.")
    else:
        print("\n  The fold does not change things much, so a left-right")
        print("  reversal inside each region is not what is missing.")
print(f"\n  -> {a.out}/fold_summary.csv")
