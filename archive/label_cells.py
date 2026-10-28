#!/usr/bin/env python
"""
Number every cell in two patches so pairs can be named, then check them.

If someone can see matching cells by eye, they can point at them. This
prints both patches with every cell numbered, so they only have to write
down pairs: "live 47 = MERSCOPE 112". Then the same script checks whether
those pairs imply a consistent offset.

That turns an argument into arithmetic:

  * if the pairs all share roughly the same offset, that offset IS the
    transform -- apply it and the matching works
  * if they imply widely different offsets, the eye is recognising
    similar-looking patterns rather than the same cells, which is easy to
    do with dense dot fields and is exactly what a displacement test is
    for

Five good pairs are enough. Ten is comfortable.

Step 1 - make the labelled figures:

    python label_cells.py --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --live-box -13200 1800 -11600 4200 --mer-box 5700 1600 7500 3600 \
        --out ./labelled

Step 2 - check the pairs they give you (a CSV with columns live_id,mer_id):

    python label_cells.py ... --pairs dirks_pairs.csv --out ./labelled
"""
import argparse, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--live-box", nargs=4, type=float, required=True)
ap.add_argument("--mer-box", nargs=4, type=float, required=True)
ap.add_argument("--flip", default="xy", help="none|x|y|xy, applied to live")
ap.add_argument("--pairs", default=None,
                help="CSV with live_id,mer_id to check")
ap.add_argument("--max-cells", type=int, default=400,
                help="refuse to label more than this - numbers stop being "
                     "readable and the exercise becomes guesswork")
ap.add_argument("--out", default="./labelled")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"

def box(df, cx, cy, b):
    x0,y0,x1,y1 = min(b[0],b[2]), min(b[1],b[3]), max(b[0],b[2]), max(b[1],b[3])
    m = (df[cx]>=x0)&(df[cx]<=x1)&(df[cy]>=y0)&(df[cy]<=y1)
    return df[m].reset_index(drop=True)

A = box(live, lx, ly, a.live_box)
B = box(mer, "center_x", "center_y", a.mer_box)
sx = -1 if "x" in a.flip else 1
sy = -1 if "y" in a.flip else 1
A = A.assign(px=A[lx]*sx, py=A[ly]*sy)
B = B.assign(px=B.center_x, py=B.center_y)
A.insert(0, "live_id", np.arange(1, len(A)+1))
B.insert(0, "mer_id",  np.arange(1, len(B)+1))
A.to_csv(f"{a.out}/live_cells_numbered.csv", index=False)
B.to_csv(f"{a.out}/mer_cells_numbered.csv", index=False)
print(f"live patch     {len(A)} cells  -> live_cells_numbered.csv")
print(f"MERSCOPE patch {len(B)} cells  -> mer_cells_numbered.csv")

# ---------------------------------------------------------------- check
if a.pairs:
    P = pd.read_csv(a.pairs)
    P = P.merge(A[["live_id","px","py"]], on="live_id") \
         .merge(B[["mer_id","px","py"]], on="mer_id", suffixes=("_l","_m"))
    if len(P) < 2:
        raise SystemExit("need at least 2 valid pairs")
    P["dx"] = P.px_m - P.px_l
    P["dy"] = P.py_m - P.py_l
    print("\n" + "="*62); print("PAIRS AS GIVEN"); print("="*62)
    print(P[["live_id","mer_id","dx","dy"]].round(1).to_string(index=False))
    sd = np.hypot(P.dx.std(), P.dy.std())
    print(f"\n  mean offset  ({P.dx.mean():+.1f}, {P.dy.mean():+.1f}) um")
    print(f"  spread       dx {P.dx.std():.1f}, dy {P.dy.std():.1f} um "
          f"(combined {sd:.1f})")
    print("\n" + "="*62); print("WHAT THIS MEANS"); print("="*62)
    if sd < 15:
        print("  The pairs agree on one offset to within a cell spacing.")
        print("  That IS the transform. Apply it and the matching works -")
        print("  the eye was right and the automated search missed it.")
    elif sd < 60:
        print("  The pairs roughly agree but scatter by more than a cell")
        print("  spacing. Could be a few mis-picked pairs among good ones:")
        print("  drop the worst and see whether the rest tighten up.")
    else:
        print("  The pairs imply completely different offsets, so they")
        print("  cannot all be correct. Moving the live data to satisfy one")
        print("  pair moves it away from the others. This is what happens")
        print("  when similar-looking local patterns are matched rather")
        print("  than the same cells.")
    P.to_csv(f"{a.out}/pairs_checked.csv", index=False)
    raise SystemExit(0)

# ---------------------------------------------------------------- draw
if len(A) > a.max_cells or len(B) > a.max_cells:
    print(f"\n  !! {max(len(A),len(B))} cells is too many to number legibly.")
    print(f"     Shrink the boxes to about {a.max_cells} cells per patch -")
    print(f"     roughly a {np.sqrt(a.max_cells/len(A))*abs(a.live_box[2]-a.live_box[0]):.0f} um window.")
    raise SystemExit(1)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def draw(ax, df, name, colour, idcol):
    """Both panels are drawn with their origin removed and the same span,
    so the two patches sit at the same scale and in the same frame. Only
    the arrangement of cells differs between them."""
    q = df.copy()
    q["qx"] = q.px - q.px.min()
    q["qy"] = q.py - q.py.min()
    ax.scatter(q.qx, q.qy, s=64, c=colour, zorder=2)
    for _, r in q.iterrows():
        ax.annotate(int(r[idcol]), (r.qx, r.qy), fontsize=6, zorder=3,
                    ha="center", va="center", color="white", weight="bold")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.set_title(f"{name} — {len(df)} cells", fontsize=14)
    ax.set_xlabel("um"); ax.grid(alpha=.25, lw=.5)
    return max(np.ptp(q.qx), np.ptp(q.qy))

# side by side, shared scale
fig, ax = plt.subplots(1, 2, figsize=(21, 11))
s1 = draw(ax[0], A, "LIVE", "#1D9E75", "live_id")
s2 = draw(ax[1], B, "MERSCOPE", "#B03A8B", "mer_id")
span = max(s1, s2) * 1.04
for x in ax:
    x.set_xlim(-span*0.02, span); x.set_ylim(span, -span*0.02)
ax[0].set_ylabel("um")
fig.suptitle("Same patch of the slide, same scale — which numbers are the same cell?",
             fontsize=16, y=0.97)
fig.tight_layout(rect=[0,0,1,0.955])
f = f"{a.out}/side_by_side_numbered.png"
fig.savefig(f, dpi=170); plt.close(fig)
print(f"  -> {f}")

# and the single panels, for zooming in on
for df, name, colour, idcol in [(A,"live","#1D9E75","live_id"),
                                (B,"merscope","#B03A8B","mer_id")]:
    fig, ax = plt.subplots(figsize=(15, 15))
    draw(ax, df, name.upper(), colour, idcol)
    ax.set_ylabel("um"); fig.tight_layout()
    f = f"{a.out}/{name}_numbered.png"
    fig.savefig(f, dpi=160); plt.close(fig)
    print(f"  -> {f}")

print("\nSend both figures. Ask for a CSV with two columns, live_id and")
print("mer_id, one row per pair they are confident about. Five is enough.")
print("Then rerun this with --pairs to check them.")
