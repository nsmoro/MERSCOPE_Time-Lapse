#!/usr/bin/env python
"""
Decode one field of view directly, and see what lowering the threshold does.

Rather than installing a pipeline built for a different file format, this
implements the decoding itself for a single field. That is enough to
answer the question being asked: does relaxing the detection threshold
recover transcripts, or is the signal simply not there?

How MERFISH decoding works, in short. Each gene has a barcode: a pattern
of on and off across the imaging rounds. Every pixel accumulates one
intensity per bit, giving it a vector. Normalise that vector and compare
it with every normalised codeword; if one is close enough, the pixel is
assigned to that gene. Neighbouring pixels sharing a gene are grouped
into a transcript.

The decisive part is the BLANK barcodes. The codebook includes patterns
deliberately assigned to no gene, so the rate at which blanks are called
measures the false-positive rate directly. Lowering the threshold always
calls more transcripts; the question is whether the blank rate rises with
them. If blanks rise as fast as genes, the extra calls are noise.

    python decode_fov.py --raw <merfish_raw_data folder> --fov 0 \
        --crop 1024 --out ./decode_fov0
"""

import argparse
import os
import re
import zipfile

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------

def read_codebook(path):
    """Codebook CSV -> (names, bit matrix, blank mask)."""
    df = pd.read_csv(path)
    bit_cols = [c for c in df.columns if re.match(r"^V\d+", str(c))]
    if not bit_cols:
        bit_cols = [c for c in df.columns if c not in ("name", "id")]
    names = df["name"].astype(str).to_numpy()
    M = df[bit_cols].to_numpy(float)
    blank = np.array([n.lower().startswith("blank") for n in names])
    print(f"  codebook: {len(names)} barcodes over {len(bit_cols)} bits "
          f"({blank.sum()} blanks, {(~blank).sum()} genes)")
    print(f"  on-bits per barcode: {int(M.sum(1).min())}-{int(M.sum(1).max())}")
    return names, M, blank, bit_cols


def read_data_organization(path):
    """dataorganization.csv -> one row per bit with round, colour, frames."""
    d = pd.read_csv(path)
    need = {"bitNumber", "imagingRound", "frame"}
    if not need <= set(d.columns):
        raise SystemExit(f"{path} lacks {need - set(d.columns)}")
    d = d.sort_values("bitNumber").reset_index(drop=True)

    def frames(v):
        if isinstance(v, str):
            return [int(float(x)) for x in re.findall(r"-?\d+\.?\d*", v)]
        return [int(v)]

    d["frames"] = d["frame"].apply(frames)
    print(f"  {len(d)} bits across rounds "
          f"{sorted(d.imagingRound.unique())}")
    return d


def load_stack_frames(zip_path, frame_ids, crop=None, offset=(0, 0)):
    """Pull specific frames out of a zipped .jp2 stack."""
    import imageio.v3 as iio
    out = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".jp2"))
        for fi in frame_ids:
            if fi >= len(names):
                continue
            with z.open(names[fi]) as fh:
                a = iio.imread(fh.read()).astype(np.float32)
            if crop:
                y, x = offset
                a = a[y:y + crop, x:x + crop]
            out.append(a)
    if not out:
        raise RuntimeError(f"no frames read from {os.path.basename(zip_path)}")
    return np.mean(out, axis=0)      # average the z planes


def high_pass(img, sigma=3.0):
    """Remove the slowly varying background that otherwise swamps spots."""
    from scipy.ndimage import gaussian_filter
    return np.maximum(img - gaussian_filter(img, sigma * 5), 0)


# ----------------------------------------------------------------------

def decode(bits, M, names, blank, dist_thresh, min_intensity,
           min_area=2, max_area=60):
    """Assign pixels to barcodes, then group them into transcripts."""
    from scipy.ndimage import label

    H, W, nb = bits.shape
    V = bits.reshape(-1, nb)
    mag = np.linalg.norm(V, axis=1)
    live = mag > min_intensity
    if live.sum() == 0:
        return pd.DataFrame(columns=["gene", "is_blank", "y", "x", "area"])

    Vn = np.zeros_like(V)
    Vn[live] = V[live] / mag[live, None]
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)

    # nearest codeword for each live pixel
    best_i = np.full(len(V), -1)
    best_d = np.full(len(V), np.inf)
    idx = np.flatnonzero(live)
    CH = 400_000
    for s in range(0, len(idx), CH):
        sl = idx[s:s + CH]
        d = 1.0 - Vn[sl] @ Mn.T              # cosine distance
        j = np.argmin(d, axis=1)
        best_i[sl] = j
        best_d[sl] = d[np.arange(len(sl)), j]

    ok = (best_d < dist_thresh) & live
    lab_img = np.where(ok, best_i, -1).reshape(H, W)

    rows = []
    for g in np.unique(lab_img):
        if g < 0:
            continue
        comp, n = label(lab_img == g)
        if n == 0:
            continue
        from scipy.ndimage import center_of_mass, sum as ndsum
        sizes = np.bincount(comp.ravel())[1:]
        keep = np.flatnonzero((sizes >= min_area) & (sizes <= max_area)) + 1
        if not len(keep):
            continue
        cms = center_of_mass(comp > 0, comp, keep)
        for (cy, cx), k in zip(np.atleast_2d(cms), keep):
            rows.append((names[g], bool(blank[g]), float(cy), float(cx),
                         int(sizes[k - 1])))
    return pd.DataFrame(rows, columns=["gene", "is_blank", "y", "x", "area"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="merfish_raw_data run folder")
    ap.add_argument("--fov", type=int, default=0)
    ap.add_argument("--crop", type=int, default=1024,
                    help="decode a crop this many pixels square (0 = all)")
    ap.add_argument("--offset", type=int, nargs=2, default=(900, 900))
    ap.add_argument("--dist", type=float, nargs="+",
                    default=[0.30, 0.40, 0.50, 0.60, 0.65, 0.70],
                    help="cosine-distance thresholds to try; higher is more "
                         "permissive")
    ap.add_argument("--min-intensity", type=float, nargs="+",
                    default=[2000, 1000, 500, 250],
                    help="intensity thresholds to try; lower is more "
                         "permissive")
    ap.add_argument("--out", default="./decode_fov")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 68)
    print("LOADING")
    print("=" * 68)
    cb = [f for f in os.listdir(a.raw) if f.startswith("codebook")]
    if not cb:
        raise SystemExit("no codebook_*.csv in the raw folder")
    names, M, blank, bit_cols = read_codebook(os.path.join(a.raw, cb[0]))
    org = read_data_organization(os.path.join(a.raw, "dataorganization.csv"))

    data_dir = os.path.join(a.raw, "data-compressed")
    if not os.path.isdir(data_dir):
        data_dir = os.path.join(a.raw, "data")
    nb = min(len(org), M.shape[1])
    print(f"\n  decoding {nb} bits of FOV {a.fov}"
          + (f", {a.crop}x{a.crop} crop at {a.offset}" if a.crop else ""))

    planes = []
    for i in range(nb):
        r = org.iloc[i]
        z = os.path.join(data_dir, f"stack_{int(r.imagingRound)}_"
                                   f"{a.fov:04d}.jp2.zip")
        if not os.path.exists(z):
            raise SystemExit(f"missing {z}")
        img = load_stack_frames(z, r.frames, a.crop or None, a.offset)
        planes.append(high_pass(img))
        print(f"    bit {i+1:>2} (round {int(r.imagingRound)}): "
              f"median {np.median(img):7.0f}  p99.9 {np.percentile(img,99.9):8.0f}")
    bits = np.stack(planes, axis=-1)
    M = M[:, :nb]

    print("\n" + "=" * 68)
    print("THRESHOLD SWEEP")
    print("=" * 68)
    print(f"{'dist':>6} {'min int':>9} {'genes':>8} {'blanks':>8} "
          f"{'blank rate':>11} {'per blank bc':>13}")
    print("-" * 62)
    rows = []
    n_blank_bc = max(int(blank.sum()), 1)
    n_gene_bc = max(int((~blank).sum()), 1)
    for di in a.dist:
        for mi in a.min_intensity:
            t = decode(bits, M, names, blank, di, mi)
            ng = int((~t.is_blank).sum()) if len(t) else 0
            nb_ = int(t.is_blank.sum()) if len(t) else 0
            rate = nb_ / max(ng + nb_, 1)
            # blanks and genes are not equally numerous, so normalise
            norm = (nb_ / n_blank_bc) / max(ng / n_gene_bc, 1e-9)
            rows.append(dict(dist=di, min_intensity=mi, genes=ng, blanks=nb_,
                             blank_rate=rate, normalised=norm))
            print(f"{di:>6.2f} {mi:>9.0f} {ng:>8d} {nb_:>8d} "
                  f"{rate:>10.1%} {norm:>12.2f}")
            if len(t):
                t.to_csv(f"{a.out}/calls_d{di:.2f}_i{mi:.0f}.csv", index=False)

    S = pd.DataFrame(rows)
    S.to_csv(f"{a.out}/threshold_sweep.csv", index=False)

    print("\n" + "=" * 68)
    print("WHAT THIS SHOWS")
    print("=" * 68)
    print("  'per blank bc' is how often a blank barcode is called relative")
    print("  to a gene barcode, adjusted for there being fewer blanks.")
    print("  A value near 1.0 means a blank is as likely to be called as a")
    print("  gene -- that is, the calls carry no real information.")
    best = S[S.genes > 0].sort_values("normalised")
    if len(best):
        b = best.iloc[0]
        print(f"\n  cleanest setting: distance {b.dist:.2f}, "
              f"intensity {b.min_intensity:.0f}")
        print(f"    {int(b.genes)} gene calls, {int(b.blanks)} blanks, "
              f"normalised rate {b.normalised:.2f}")
        if b.normalised > 0.5:
            print("\n  Even at its best, blanks are called nearly as often as")
            print("  genes. Lowering the threshold further adds noise rather")
            print("  than signal.")
    else:
        print("\n  Nothing was called at any setting tried.")
    print(f"\n  -> {a.out}/threshold_sweep.csv")


if __name__ == "__main__":
    main()
