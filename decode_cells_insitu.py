#!/usr/bin/env python
"""
Decoded calls assigned to cells segmented from the same image crop.

Assigning calls to the vendor's cell table means converting from camera
pixels to stage coordinates to mosaic microns, and each of those steps is
a place to be quietly wrong. This avoids all of it: cells are segmented
from the DAPI and PolyT frames of the SAME crop that was decoded, so
calls and cells share one coordinate system and no conversion happens.

For the question being asked -- how are genes and blanks distributed over
single cells, and is there a threshold window where genes appear without
blanks -- it makes no difference whose segmentation defines the cells.

Two statistics distinguish the possibilities:

  share of cells with no blanks   uniform background leaves few cells
                                  clean; a misbehaving subpopulation
                                  leaves most cells clean
  blanks in the worst 10% of cells  near 10-25% means evenly spread;
                                  much higher means concentrated in a
                                  few cells, which would fit nonspecific
                                  binding

    python decode_cells_insitu.py --raw <merfish_raw_data run folder> \
        --fov 0 --crop 1024 --out ./insitu
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_fov import (read_codebook, read_data_organization,
                        load_stack_frames, high_pass, decode)
import re as _re


def read_data_organization_raw(path):
    """The organization table unfiltered, so the stain rows survive."""
    d = pd.read_csv(path)

    def frames(v):
        if isinstance(v, str):
            return [int(float(x)) for x in _re.findall(r"-?\d+\.?\d*", v)]
        return [int(v)]

    d["frames"] = d["frame"].apply(frames)
    return d


def segment_crop(img, px_um, diameter_um=9.0, gpu=True):
    """Cells from the crop itself. Cellpose if available, else a
    threshold-and-watershed fallback so the script still runs."""
    try:
        from cellpose import models
        import torch
        g = gpu and torch.cuda.is_available()
        m = models.CellposeModel(gpu=g)
        masks = m.eval(img, diameter=diameter_um / px_um, normalize=True)[0]
        if masks.max() > 0:
            print(f"  Cellpose found {masks.max()} cells "
                  f"({'GPU' if g else 'CPU'})")
            return masks
        print("  Cellpose found nothing; falling back to thresholding")
    except Exception as e:
        print(f"  Cellpose unavailable ({type(e).__name__}); thresholding")

    from skimage.filters import threshold_otsu, gaussian
    from skimage.measure import label
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.segmentation import watershed
    from scipy import ndimage as ndi

    s = gaussian(img.astype(float), 1.5)
    bw = s > threshold_otsu(s)
    bw = binary_opening(bw, disk(2))
    bw = remove_small_objects(bw, int(20 / px_um ** 2))
    dist = ndi.distance_transform_edt(bw)
    from skimage.feature import peak_local_max
    pk = peak_local_max(dist, min_distance=int(max(diameter_um / px_um / 2, 3)),
                        labels=bw)
    mk = np.zeros(dist.shape, int)
    mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
    masks = watershed(-dist, mk, mask=bw)
    print(f"  thresholding found {masks.max()} cells")
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--offset", type=int, nargs=2, default=(900, 900))
    ap.add_argument("--px-um", type=float, default=0.108)
    ap.add_argument("--cell-diameter-um", type=float, default=9.0)
    ap.add_argument("--seg-channel", default="auto",
                    help="auto | dapi | polyt | sum — which frames to segment")
    ap.add_argument("--dist", type=float, nargs="+",
                    default=[0.35, 0.40, 0.44, 0.48, 0.50, 0.55, 0.60, 0.65])
    ap.add_argument("--min-intensity", type=float, nargs="+",
                    default=[2000, 1000, 500])
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="./insitu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 74)
    print("LOADING")
    print("=" * 74)
    cb = [f for f in os.listdir(a.raw) if f.startswith("codebook")]
    names, M, blank, _ = read_codebook(os.path.join(a.raw, cb[0]))
    org = read_data_organization(os.path.join(a.raw, "dataorganization.csv"))
    nb = min(len(org), M.shape[1]); M = M[:, :nb]

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
    print(f"  decoded field {a.fov}, {a.crop}x{a.crop} crop, {nb} bits\n")

    # the stain channels live on their own row of the organization table,
    # with no bit number. DAPI marks nuclei and is the right thing to
    # segment -- averaging the barcode frames gives a composite that means
    # nothing and that Cellpose cannot work with.
    full = read_data_organization_raw(os.path.join(a.raw,
                                                   "dataorganization.csv"))
    stain = {str(r.channelName).lower(): r for _, r in full.iterrows()
             if pd.isna(r.bitNumber)}
    want = a.seg_channel.lower()
    pick = None
    for key in ([want] if want != "auto" else ["dapi", "polyt"]):
        if key in stain:
            pick = stain[key]; break
    if pick is None:
        raise SystemExit(f"no stain channel to segment; table has "
                         f"{sorted(stain)}")
    z = os.path.join(data_dir,
                     f"stack_{int(pick.imagingRound)}_{a.fov:04d}.jp2.zip")
    if not os.path.exists(z):
        alt = [f for f in os.listdir(data_dir)
               if f.endswith(f"_{a.fov:04d}.jp2.zip")]
        raise SystemExit(f"no stack for round {int(pick.imagingRound)}; "
                         f"available: {sorted(alt)[:6]}")
    cell_img = load_stack_frames(z, pick.frames, a.crop or None, a.offset)
    print(f"  segmenting on {pick.channelName} "
          f"(round {int(pick.imagingRound)}, frames {pick.frames}): "
          f"median {np.median(cell_img):.0f}, "
          f"p99 {np.percentile(cell_img, 99):.0f}")
    print("SEGMENTING THE SAME CROP")
    masks = segment_crop(cell_img, a.px_um, a.cell_diameter_um,
                         gpu=not a.cpu)
    n_cells = int(masks.max())
    if n_cells == 0:
        raise SystemExit("no cells segmented in this crop; try another FOV "
                         "or offset")
    area_mm2 = (a.crop * a.px_um) ** 2 / 1e6
    dens = n_cells / area_mm2
    print(f"  {n_cells} cells in {area_mm2:.4f} mm2 ({dens:.0f} per mm2)")
    if dens > 600 or dens < 20:
        print(f"  !! that density is implausible for these cells "
              f"(~180/mm2 elsewhere on this slide).")
        print(f"     Per-cell numbers below will be unreliable; the objects")
        print(f"     found are probably debris rather than cells.")
    print()

    print("=" * 74)
    print("THRESHOLD SWEEP, PER CELL")
    print("=" * 74)
    print(f"{'dist':>6} {'min int':>8} {'genes':>8} {'blanks':>7} "
          f"{'in cells':>9} {'g/cell':>8} {'b/cell':>8} "
          f"{'no blanks':>10} {'worst 10%':>10}")
    print("-" * 86)

    rows, store = [], {}
    for di in a.dist:
        for mi in a.min_intensity:
            t = decode(bits, M, names, blank, di, mi)
            if not len(t):
                print(f"{di:>6.2f} {mi:>8.0f} {0:>8} {0:>7}   nothing called")
                rows.append(dict(dist=di, min_intensity=mi, genes=0, blanks=0,
                                 in_cells=0))
                continue
            yy = np.clip(t.y.to_numpy().astype(int), 0, masks.shape[0]-1)
            xx = np.clip(t.x.to_numpy().astype(int), 0, masks.shape[1]-1)
            t = t.assign(cell=masks[yy, xx])
            inside = t[t.cell > 0]
            G = np.zeros(n_cells + 1); B = np.zeros(n_cells + 1)
            if len(inside):
                gg = inside[~inside.is_blank].groupby("cell").size()
                bb = inside[inside.is_blank].groupby("cell").size()
                G[gg.index] = gg.to_numpy(); B[bb.index] = bb.to_numpy()
            G, B = G[1:], B[1:]
            frac0 = float((B == 0).mean())
            tail = (float(np.sort(B)[::-1][:max(1, n_cells // 10)].sum()
                          / max(B.sum(), 1)) if B.sum() else 0.0)
            print(f"{di:>6.2f} {mi:>8.0f} {int((~t.is_blank).sum()):>8d} "
                  f"{int(t.is_blank.sum()):>7d} {len(inside):>9d} "
                  f"{G.mean():>8.2f} {B.mean():>8.2f} {frac0:>9.1%} "
                  f"{tail:>9.1%}")
            rows.append(dict(dist=di, min_intensity=mi,
                             genes=int((~t.is_blank).sum()),
                             blanks=int(t.is_blank.sum()),
                             in_cells=len(inside), g_per_cell=float(G.mean()),
                             b_per_cell=float(B.mean()),
                             frac_zero_blank=frac0, worst10=tail))
            store[(di, mi)] = (G, B)
            inside.to_csv(f"{a.out}/calls_d{di:.2f}_i{mi:.0f}.csv", index=False)

    S = pd.DataFrame(rows); S.to_csv(f"{a.out}/sweep.csv", index=False)

    print("\n" + "=" * 74)
    print("IS THERE A WINDOW WITH GENES BUT NO BLANKS?")
    print("=" * 74)
    clean = S[(S.genes > 0) & (S.blanks == 0)]
    if len(clean):
        print("  Yes:")
        print(clean[["dist", "min_intensity", "genes", "in_cells"]]
              .to_string(index=False))
    else:
        nz = S[S.genes > 0]
        if len(nz):
            ratio = nz.blanks / nz.genes.clip(lower=1)
            b = nz.loc[ratio.idxmin()]
            n_bl = max(int(blank.sum()), 1)
            n_gn = max(int((~blank).sum()), 1)
            exp = n_gn / n_bl
            obs = b.genes / max(b.blanks, 1)
            print(f"  No. The cleanest setting is distance {b.dist:.2f}, "
                  f"intensity {b.min_intensity:.0f}:")
            print(f"    {int(b.genes)} genes, {int(b.blanks)} blanks "
                  f"= {obs:.1f} genes per blank")
            print(f"    pure chance would give about {exp:.1f}, since there "
                  f"are {n_gn} gene barcodes to {n_bl} blanks")
            print(f"    -> {'worse than' if obs < exp else 'better than'} "
                  f"chance")
        else:
            print("  Nothing called anywhere in this range.")

    print("\n" + "=" * 74)
    print("ARE BLANKS CONCENTRATED IN A FEW CELLS?")
    print("=" * 74)
    hasb = S[S.get("blanks", 0) > 0] if "blanks" in S else S.iloc[0:0]
    if len(hasb):
        w = hasb.worst10.median(); z = hasb.frac_zero_blank.median()
        print(f"  worst 10% of cells carry {w:.0%} of blanks (median across "
              f"settings)")
        print(f"  {z:.0%} of cells have no blank calls")
        if w > 0.5 and z > 0.7:
            print("\n  Concentrated. Most cells are clean and a few carry")
            print("  the blanks, which fits a subpopulation binding probes")
            print("  nonspecifically. Worth looking at what those cells are.")
        else:
            print("\n  Spread fairly evenly, which points to uniform")
            print("  background rather than a particular subpopulation.")
    else:
        print("  No blanks called, so nothing to distribute.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in sorted(store) if store[k][0].sum() + store[k][1].sum() > 0]
    if keys:
        keys = keys[:6]
        fig, ax = plt.subplots(2, len(keys), figsize=(4.2*len(keys), 8),
                               squeeze=False)
        for k, key in enumerate(keys):
            G, B = store[key]
            ax[0][k].hist(G, bins=range(0, int(max(G.max(), 1)) + 2),
                          color="#1D9E75")
            ax[0][k].set_title(f"d={key[0]:.2f} i={key[1]:.0f}\n"
                               f"genes per cell (mean {G.mean():.1f})",
                               fontsize=10)
            ax[1][k].hist(B, bins=range(0, int(max(B.max(), 1)) + 2),
                          color="#B03A8B")
            ax[1][k].set_title(f"blanks per cell (mean {B.mean():.2f})\n"
                               f"{(B==0).mean():.0%} of cells have none",
                               fontsize=10)
            for A_ in (ax[0][k], ax[1][k]):
                A_.set_xlabel("calls per cell"); A_.set_ylabel("cells")
        fig.suptitle(f"Per-cell distributions, field {a.fov} "
                     f"({n_cells} cells segmented from the same crop)",
                     fontsize=14)
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
        fig.suptitle("Do genes and blanks rise together?", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{a.out}/genes_vs_blanks.png", dpi=145); plt.close(fig)
        print(f"  -> {a.out}/genes_vs_blanks.png")

    np.save(f"{a.out}/cell_masks.npy", masks)
    print(f"  -> {a.out}/sweep.csv")


if __name__ == "__main__":
    main()
