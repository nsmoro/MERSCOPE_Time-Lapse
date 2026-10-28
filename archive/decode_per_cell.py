#!/usr/bin/env python
"""
Decoded calls assigned to single cells, across a fine range of thresholds.

The earlier sweep pooled every call in an image crop, which cannot tell a
uniform low-level false-positive rate apart from a subpopulation of cells
binding probes nonspecifically. This assigns each call to a cell and looks
at the distributions.

Two things it answers:

  * do genes and blanks move together as the threshold relaxes, or is
    there a window where genes appear while blanks stay at zero?
  * are blanks spread evenly across cells, or concentrated in a few? A
    Poisson-like spread points to uniform background; a long tail with
    most cells at zero points to a subpopulation behaving differently.

Calls are decoded in field-of-view pixel coordinates, converted to
microns using the field's stage position, and assigned to the nearest
MERSCOPE cell centroid within a radius.

    python decode_per_cell.py --raw <merfish_raw_data run folder> \
        --merscope-region <.../region_R9> --fov 0 --out ./per_cell
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_fov import (read_codebook, read_data_organization,
                        load_stack_frames, high_pass, decode)


def read_positions(raw, fov):
    """FOV stage position in microns, from positions.csv."""
    for name in ("positions.csv", "settings/positions.csv"):
        p = os.path.join(raw, name)
        if os.path.exists(p):
            try:
                d = pd.read_csv(p, header=None)
            except Exception:
                continue
            if len(d) > fov and d.shape[1] >= 2:
                return float(d.iloc[fov, 0]), float(d.iloc[fov, 1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--merscope-region", required=True,
                    help="folder with cell_metadata.csv for the region this "
                         "field sits in")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--offset", type=int, nargs=2, default=(900, 900))
    ap.add_argument("--px-um", type=float, default=0.108,
                    help="camera pixel size in microns")
    ap.add_argument("--assign-um", type=float, default=12.0,
                    help="a call is assigned to a cell centroid within this "
                         "distance")
    ap.add_argument("--dist", type=float, nargs="+",
                    default=[0.35, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50],
                    help="finer steps between strict and medium")
    ap.add_argument("--min-intensity", type=float, nargs="+",
                    default=[3000, 2000, 1500, 1000])
    ap.add_argument("--out", default="./per_cell")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 70)
    print("LOADING")
    print("=" * 70)
    cb = [f for f in os.listdir(a.raw) if f.startswith("codebook")]
    names, M, blank, _ = read_codebook(os.path.join(a.raw, cb[0]))
    org = read_data_organization(os.path.join(a.raw, "dataorganization.csv"))
    nb = min(len(org), M.shape[1]); M = M[:, :nb]

    cells = pd.read_csv(os.path.join(a.merscope_region, "cell_metadata.csv"))
    cx = "center_x" if "center_x" in cells else cells.columns[1]
    cy = "center_y" if "center_y" in cells else cells.columns[2]
    C = cells[[cx, cy]].to_numpy(float)
    print(f"  {len(C):,} cells in the region")

    pos = read_positions(a.raw, a.fov)
    print("  field position: " + (str(pos) if pos else
          "not found - anchoring on the region's cells instead"))

    data_dir = os.path.join(a.raw, "data-compressed")
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(a.raw, "data")
    planes = []
    for i in range(nb):
        r = org.iloc[i]
        z = os.path.join(data_dir,
                         f"stack_{int(r.imagingRound)}_{a.fov:04d}.jp2.zip")
        planes.append(high_pass(load_stack_frames(z, r.frames,
                                                  a.crop or None, a.offset)))
    bits = np.stack(planes, axis=-1)
    print(f"  decoded field {a.fov}, {a.crop}x{a.crop} crop\n")

    from scipy.spatial import cKDTree
    tree = cKDTree(C)

    def to_um(df):
        """call pixel coords -> microns in the region's frame"""
        x = (df.x.to_numpy() + a.offset[1]) * a.px_um
        y = (df.y.to_numpy() + a.offset[0]) * a.px_um
        if pos:
            x = x + pos[0]; y = y + pos[1]
        else:                       # no stage position: anchor on the cells
            x = x - x.mean() + C[:, 0].mean()
            y = y - y.mean() + C[:, 1].mean()
        return np.c_[x, y]

    print("=" * 70)
    print("THRESHOLD SWEEP, PER CELL")
    print("=" * 70)
    print(f"{'dist':>6} {'min int':>8} {'genes':>8} {'blanks':>7} "
          f"{'cells hit':>10} {'g/cell':>8} {'b/cell':>8} "
          f"{'cells 0 blank':>14} {'blank tail':>11}")
    print("-" * 90)

    rows, per_cell_store = [], {}
    for di in a.dist:
        for mi in a.min_intensity:
            t = decode(bits, M, names, blank, di, mi)
            if not len(t):
                print(f"{di:>6.2f} {mi:>8.0f} {0:>8} {0:>7}      nothing called")
                rows.append(dict(dist=di, min_intensity=mi, genes=0, blanks=0,
                                 cells=0, g_per_cell=0, b_per_cell=0,
                                 frac_zero_blank=np.nan, tail=np.nan))
                continue
            P = to_um(t)
            d, j = tree.query(P, k=1)
            ok = d < a.assign_um
            t = t.assign(cell=np.where(ok, j, -1))
            t = t[t.cell >= 0]
            if not len(t):
                print(f"{di:>6.2f} {mi:>8.0f} {0:>8} {0:>7}      "
                      f"none within {a.assign_um:.0f} um of a cell")
                continue

            g = t[~t.is_blank].groupby("cell").size()
            b = t[t.is_blank].groupby("cell").size()
            hit = sorted(set(g.index) | set(b.index))
            G = g.reindex(hit).fillna(0).to_numpy()
            B = b.reindex(hit).fillna(0).to_numpy()
            frac0 = float((B == 0).mean())
            # how concentrated are the blanks? share carried by the top 10%
            tail = (float(np.sort(B)[::-1][:max(1, len(B)//10)].sum() /
                          max(B.sum(), 1)) if B.sum() else np.nan)
            print(f"{di:>6.2f} {mi:>8.0f} {int(G.sum()):>8d} "
                  f"{int(B.sum()):>7d} {len(hit):>10d} {G.mean():>8.2f} "
                  f"{B.mean():>8.2f} {frac0:>13.1%} {tail:>10.1%}")
            rows.append(dict(dist=di, min_intensity=mi, genes=int(G.sum()),
                             blanks=int(B.sum()), cells=len(hit),
                             g_per_cell=float(G.mean()),
                             b_per_cell=float(B.mean()),
                             frac_zero_blank=frac0, tail=tail))
            per_cell_store[(di, mi)] = (G, B)
            t.to_csv(f"{a.out}/calls_d{di:.2f}_i{mi:.0f}.csv", index=False)

    S = pd.DataFrame(rows); S.to_csv(f"{a.out}/sweep_per_cell.csv", index=False)

    print("\n" + "=" * 70)
    print("IS THERE A WINDOW WITH GENES BUT NO BLANKS?")
    print("=" * 70)
    clean = S[(S.genes > 0) & (S.blanks == 0)]
    if len(clean):
        print("  Yes — settings calling genes with no blanks at all:")
        print(clean[["dist", "min_intensity", "genes", "cells",
                     "g_per_cell"]].to_string(index=False))
    else:
        nz = S[S.genes > 0]
        if len(nz):
            best = nz.loc[(nz.blanks / nz.genes.clip(lower=1)).idxmin()]
            print(f"  No. The cleanest setting still calls blanks: "
                  f"distance {best.dist:.2f}, intensity "
                  f"{best.min_intensity:.0f} gives {int(best.genes)} genes "
                  f"and {int(best.blanks)} blanks.")
            print("  Genes and blanks appear together at every threshold.")
        else:
            print("  Nothing was called at any setting in this range.")

    print("\n" + "=" * 70)
    print("ARE BLANKS CONCENTRATED IN SOME CELLS?")
    print("=" * 70)
    print("  'cells 0 blank' is the share of cells with no blank calls.")
    print("  'blank tail' is the share of all blanks carried by the worst")
    print("  10% of cells. Near 10% means blanks are spread evenly, which")
    print("  points to uniform background. Much higher means a subset of")
    print("  cells carries them, which would fit nonspecific binding in a")
    print("  subpopulation.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted(per_cell_store)[:6]
    if keys:
        fig, ax = plt.subplots(2, len(keys), figsize=(4.2*len(keys), 8),
                               squeeze=False)
        for k, key in enumerate(keys):
            G, B = per_cell_store[key]
            ax[0][k].hist(G, bins=range(0, int(max(G.max(), 1)) + 2),
                          color="#1D9E75")
            ax[0][k].set_title(f"d={key[0]:.2f}, i={key[1]:.0f}\n"
                               f"genes per cell", fontsize=10)
            ax[1][k].hist(B, bins=range(0, int(max(B.max(), 1)) + 2),
                          color="#B03A8B")
            ax[1][k].set_title(f"blanks per cell\n"
                               f"{(B==0).mean():.0%} of cells have none",
                               fontsize=10)
            for A_ in (ax[0][k], ax[1][k]):
                A_.set_xlabel("per cell"); A_.set_ylabel("cells")
        fig.suptitle("Per-cell distributions across thresholds — "
                     "genes above, blanks below", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{a.out}/per_cell_distributions.png", dpi=145)
        plt.close(fig)
        print(f"\n  -> {a.out}/per_cell_distributions.png")

    nz = S[S.genes > 0]
    if len(nz) > 1:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        for mi, g in nz.groupby("min_intensity"):
            g = g.sort_values("dist")
            ax[0].plot(g.dist, g.genes, "o-", label=f"intensity {mi:.0f}")
            ax[1].plot(g.dist, g.blanks, "o-", label=f"intensity {mi:.0f}")
        ax[0].set_title("gene calls"); ax[1].set_title("blank calls")
        for A_ in ax:
            A_.set_xlabel("distance threshold (more permissive to the right)")
            A_.set_ylabel("calls"); A_.legend(fontsize=9); A_.grid(alpha=.25)
        fig.suptitle("Do genes and blanks move together?", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{a.out}/genes_vs_blanks.png", dpi=145); plt.close(fig)
        print(f"  -> {a.out}/genes_vs_blanks.png")

    print(f"  -> {a.out}/sweep_per_cell.csv")


if __name__ == "__main__":
    main()
