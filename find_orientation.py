#!/usr/bin/env python
"""
Work out which orientation relates two datasets, before matching anything.

Run this first. Matching cells is meaningless until the two coordinate
systems are in the same orientation, and the answer is not guessable: a
slide can go into an instrument eight ways, and the software on either
side may flip or rotate the coordinates it reports.

The test uses BLOCKS rather than cells. Each dataset is divided into its
natural groups -- wells, regions, fields -- and the question asked is: in
which orientation do the block centres line up consistently? If an
orientation is correct, every block pair implies nearly the same offset.
If it is wrong, the implied offsets scatter.

That works even when individual cells do not correspond at all, which is
what makes it a sound first step: it establishes the orientation without
assuming the thing you are trying to find out.

    python find_orientation.py \
        --live live_coordinates.csv --merscope merscope_cells.csv \
        --region-order R9,R11,R12,R13 --out ./orientation

With --mapping you can test a specific block-to-block correspondence; by
default blocks are paired in the order given.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd


OPS = {
    "none":   (lambda P: P.copy()),
    "x":      (lambda P: P * [-1, 1]),
    "y":      (lambda P: P * [1, -1]),
    "xy":     (lambda P: P * [-1, -1]),
    "swap":   (lambda P: P[:, ::-1].copy()),
    "swap_x": (lambda P: P[:, ::-1] * [-1, 1]),
    "swap_y": (lambda P: P[:, ::-1] * [1, -1]),
    "swap_xy":(lambda P: P[:, ::-1] * [-1, -1]),
}

LABEL = {
    "none": "as acquired", "x": "mirrored left-right",
    "y": "mirrored top-bottom", "xy": "rotated 180°",
    "swap": "transposed", "swap_x": "rotated 90°",
    "swap_y": "rotated 270°", "swap_xy": "transposed and rotated 180°",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--merscope", required=True)
    ap.add_argument("--region-order", required=True,
                    help="MERSCOPE regions in the order they correspond to "
                         "the live positions")
    ap.add_argument("--per-region", type=int, default=3,
                    help="how many consecutive live positions per region")
    ap.add_argument("--groups", default=None,
                    help="explicit mapping, e.g. R9:1-3,R11:4-6,R20:25-36")
    ap.add_argument("--out", default="./orientation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
    lx = "x_um" if "x_um" in live else "CentroidXglobal"
    ly = "y_um" if "y_um" in live else "CentroidYglobal"
    regions = [r.strip() for r in a.region_order.split(",")]
    positions = sorted(live.Position.unique())

    if a.groups:
        groups = {}
        for part in a.groups.split(","):
            reg, rng = part.split(":")
            lo, hi = (int(v) for v in rng.split("-"))
            groups[reg.strip()] = list(range(lo, hi + 1))
    else:
        groups = {}
        for k, reg in enumerate(regions):
            lo = k * a.per_region
            hi = ((k + 1) * a.per_region if k < len(regions) - 1
                  else len(positions))
            groups[reg] = positions[lo:hi]

    print("BLOCK PAIRS UNDER TEST")
    pairs = []
    for reg, ps in groups.items():
        s = live[live.Position.isin(ps)]
        g = mer[mer.region == reg]
        if len(s) < 10 or len(g) < 10:
            print(f"  {reg:>6}  skipped, too few cells")
            continue
        pairs.append((reg, s[[lx, ly]].to_numpy(float),
                      g[["center_x", "center_y"]].to_numpy(float)))
        print(f"  {reg:>6}  <-  positions {ps[0]}..{ps[-1]}   "
              f"{len(s):>5} live, {len(g):>5} MERSCOPE")
    if len(pairs) < 3:
        raise SystemExit("\nneed at least three block pairs to compare "
                         "orientations")

    print(f"\n{'orientation':<26} {'dx spread':>11} {'dy spread':>11} "
          f"{'combined':>11}")
    print("-" * 62)
    rows = []
    for name, op in OPS.items():
        offs = []
        for _, A, B in pairs:
            offs.append(B.mean(0) - op(A).mean(0))
        O = np.array(offs)
        sx, sy = float(O[:, 0].std()), float(O[:, 1].std())
        rows.append(dict(orientation=name, label=LABEL[name],
                         dx_spread=sx, dy_spread=sy,
                         combined=float(np.hypot(sx, sy)),
                         dx_mean=float(O[:, 0].mean()),
                         dy_mean=float(O[:, 1].mean())))
        print(f"{name + '  (' + LABEL[name] + ')':<26} "
              f"{sx:>10.0f}u {sy:>10.0f}u {np.hypot(sx, sy):>10.0f}u")

    S = pd.DataFrame(rows).sort_values("combined").reset_index(drop=True)
    S.to_csv(f"{a.out}/orientation_test.csv", index=False)
    best, second = S.iloc[0], S.iloc[1]
    margin = second.combined / max(best.combined, 1e-9)

    print("\n" + "=" * 62)
    print("WHICH ORIENTATION?")
    print("=" * 62)
    print(f"  best    {best.orientation}  ({best.label})")
    print(f"          spread {best.combined:.0f} um, "
          f"offset ({best.dx_mean:+.0f}, {best.dy_mean:+.0f}) um")
    print(f"  next    {second.orientation}  ({second.label})")
    print(f"          spread {second.combined:.0f} um")
    print(f"\n  the best fits {margin:.1f}x better than the next")
    if margin > 5:
        print("\n  Clear. Use --flip " + best.orientation + " in the "
              "matching scripts.")
        print("  Every block agrees on one offset, which cannot happen in")
        print("  the wrong orientation.")
    elif margin > 2:
        print("\n  Probably right, but not decisively. Worth checking the")
        print("  figure, and confirming against the bench record if there")
        print("  is one.")
    else:
        print("\n  No orientation stands out. Either the block mapping is")
        print("  wrong -- check which block corresponds to which -- or the")
        print("  blocks are too symmetric to tell them apart.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(OPS)
    fig, ax = plt.subplots(2, (n + 1) // 2 + 1,
                           figsize=(3.4 * ((n + 1) // 2 + 1), 7.4))
    ax = ax.ravel()
    B_all = np.vstack([B for _, _, B in pairs])
    cmap = plt.get_cmap("tab10")
    a0 = ax[0]
    for k, (reg, _, B) in enumerate(pairs):
        a0.scatter(B[:, 0], B[:, 1], s=1.2, color=cmap(k % 10), linewidths=0)
    a0.set_aspect("equal"); a0.invert_yaxis()
    a0.set_xticks([]); a0.set_yticks([])
    a0.set_title("MERSCOPE (reference)", fontsize=11, weight="bold")
    for i, (name, op) in enumerate(OPS.items(), start=1):
        A_ = ax[i]
        for k, (reg, A, _) in enumerate(pairs):
            P = op(A)
            A_.scatter(P[:, 0], P[:, 1], s=1.2, color=cmap(k % 10),
                       linewidths=0)
        A_.set_aspect("equal"); A_.invert_yaxis()
        A_.set_xticks([]); A_.set_yticks([])
        sp = S[S.orientation == name].combined.iloc[0]
        is_best = name == best.orientation
        A_.set_title(f"live: {LABEL[name]}\nspread {sp:.0f} um",
                     fontsize=10,
                     color="#1A7A4C" if is_best else "#444",
                     weight="bold" if is_best else "normal")
        if is_best:
            for s_ in A_.spines.values():
                s_.set_color("#1A7A4C"); s_.set_linewidth(2.5)
    for j in range(len(OPS) + 1, len(ax)):
        ax[j].axis("off")
    fig.suptitle("Which orientation puts the blocks in corresponding "
                 "places?\nmatching colours are the same block; the lowest "
                 "spread wins", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    f = f"{a.out}/orientations.png"
    fig.savefig(f, dpi=145); plt.close(fig)
    print(f"\n  -> {f}")
    print(f"  -> {a.out}/orientation_test.csv")


if __name__ == "__main__":
    main()
