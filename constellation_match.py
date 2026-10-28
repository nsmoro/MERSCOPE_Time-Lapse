#!/usr/bin/env python
"""
Match cells by the pattern their neighbours make, not by where they sit.

Everything tried so far assumed one transform maps the whole slide. This
does not. It describes each cell by the SHAPE formed by its nearest
neighbours, in a way that does not change when the slide is shifted,
rotated or scaled, then looks for cells whose neighbourhoods have the
same shape.

The method is the one astronomers use to work out where a telescope was
pointing with no prior information. Take four nearby cells. Use the two
furthest apart to define a local frame, and express the other two in that
frame. The resulting four numbers are a fingerprint of that little
constellation: the same four cells produce the same fingerprint in both
datasets however the slide is oriented.

Matching fingerprints is only the first half. Any two dense point sets
share some fingerprints by accident, so each match is treated as a VOTE
for a particular transform. If the cells really correspond, many
independent constellations vote for the SAME transform, and that
agreement is what cannot happen by chance. A shuffled control shows how
much agreement arises from nothing.

    python constellation_match.py \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --region R18 --positions 22,23,24 --out ./constellation
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------
# fingerprints
# ----------------------------------------------------------------------

def quad_hash(p):
    """Fingerprint of four points, unchanged by shift, rotation or scale.

    The two furthest-apart points become (0,0) and (1,1); the other two
    are reported in that frame. Returns None when the quad is degenerate.
    """
    best, pair = -1.0, None
    for i, j in itertools.combinations(range(4), 2):
        d = np.hypot(*(p[i] - p[j]))
        if d > best:
            best, pair = d, (i, j)
    if best <= 0:
        return None
    i, j = pair
    a, b = p[i], p[j]
    v = b - a
    scale = np.dot(v, v)
    if scale <= 0:
        return None
    # rotate so a->b is the diagonal, scale so |a->b| = 1
    rot = np.array([[v[0], v[1]], [-v[1], v[0]]]) / np.sqrt(scale)
    rest = [k for k in range(4) if k not in (i, j)]
    out = []
    for k in rest:
        q = rot @ (p[k] - a) / np.sqrt(scale)
        out.extend(q)
    # the two remaining points are interchangeable: order them so the
    # same quad always gives the same fingerprint
    c, d = np.array(out[:2]), np.array(out[2:])
    if (c[0], c[1]) > (d[0], d[1]):
        c, d = d, c
    h = np.concatenate([c, d])
    # a quad and its mirror image are different constellations; keep the
    # orientation so a reflected layout does not match a normal one
    return h, (i, j, rest[0], rest[1])


def build_quads(P, k_neighbours=6, max_per_cell=4, seed=0):
    """Fingerprints for the small constellations around each cell."""
    if len(P) < 5:
        return np.empty((0, 4)), []
    tree = cKDTree(P)
    k = min(k_neighbours + 1, len(P))
    _, idx = tree.query(P, k=k)
    rng = np.random.default_rng(seed)
    H, members = [], []
    for row in idx:
        centre, nbrs = row[0], row[1:]
        combos = list(itertools.combinations(nbrs, 3))
        if len(combos) > max_per_cell:
            sel = rng.choice(len(combos), max_per_cell, replace=False)
            combos = [combos[s] for s in sel]
        for c in combos:
            ids = np.array([centre, *c])
            r = quad_hash(P[ids])
            if r is None:
                continue
            h, order = r
            if not np.all(np.isfinite(h)):
                continue
            H.append(h)
            members.append(ids[list(order)])
    return np.asarray(H), members


# ----------------------------------------------------------------------
# voting
# ----------------------------------------------------------------------

def similarity_from_pairs(src, dst):
    """Best shift+rotation+scale taking src onto dst (4 point pairs)."""
    cs, cd = src.mean(0), dst.mean(0)
    s0, d0 = src - cs, dst - cd
    n_s = np.sqrt((s0 ** 2).sum())
    if n_s < 1e-9:
        return None
    U, S, Vt = np.linalg.svd(d0.T @ s0)
    R = U @ Vt
    if np.linalg.det(R) < 0:                 # keep it a rotation
        U[:, -1] *= -1
        R = U @ Vt
    scale = S.sum() / (n_s ** 2)
    t = cd - scale * (R @ cs)
    ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return scale, ang, t


def vote(A, B, hA, mA, hB, mB, hash_tol, scale_range, out_prefix=None):
    """Match fingerprints, then see whether the matches agree on a transform."""
    if len(hA) == 0 or len(hB) == 0:
        return dict(n_quad_matches=0, n_consistent=0, best=None, votes=None)
    tree = cKDTree(hB)
    pairs = tree.query_ball_point(hA, r=hash_tol)
    votes = []
    for ia, js in enumerate(pairs):
        for jb in js:
            r = similarity_from_pairs(A[mA[ia]], B[mB[jb]])
            if r is None:
                continue
            scale, ang, t = r
            if not (scale_range[0] < scale < scale_range[1]):
                continue
            votes.append((scale, ang, t[0], t[1], ia, jb))
    if not votes:
        return dict(n_quad_matches=0, n_consistent=0, best=None, votes=None)
    V = np.array([v[:4] for v in votes], float)

    # how many votes agree with each vote, within a tolerance?
    # agreement in rotation and translation is what matters
    tol_ang, tol_t = 2.0, 60.0
    t_tree = cKDTree(V[:, 2:4])
    best_n, best_i = 0, 0
    for i in range(len(V)):
        near = t_tree.query_ball_point(V[i, 2:4], r=tol_t)
        if not near:
            continue
        da = np.abs(((V[near, 1] - V[i, 1] + 180) % 360) - 180)
        n = int((da < tol_ang).sum())
        if n > best_n:
            best_n, best_i = n, i
    return dict(n_quad_matches=len(votes), n_consistent=best_n,
                best=V[best_i], votes=V)


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--merscope", required=True)
    ap.add_argument("--region", default=None)
    ap.add_argument("--positions", default=None)
    ap.add_argument("--box", nargs=4, type=float, default=None,
                    help="restrict the live cells to x0 y0 x1 y1")
    ap.add_argument("--k-neighbours", type=int, default=6)
    ap.add_argument("--max-per-cell", type=int, default=4)
    ap.add_argument("--hash-tol", type=float, default=0.02,
                    help="how close two fingerprints must be to count as "
                         "the same constellation")
    ap.add_argument("--scale-min", type=float, default=0.9)
    ap.add_argument("--scale-max", type=float, default=1.1)
    ap.add_argument("--n-null", type=int, default=5)
    ap.add_argument("--out", default="./constellation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
    lx = "x_um" if "x_um" in live else "CentroidXglobal"
    ly = "y_um" if "y_um" in live else "CentroidYglobal"
    if a.positions:
        ps = [int(p) for p in a.positions.split(",")]
        live = live[live.Position.isin(ps)]
    A = live[[lx, ly]].to_numpy(float)
    if a.box:
        x0, y0, x1, y1 = a.box
        m = ((A[:, 0] >= min(x0, x1)) & (A[:, 0] <= max(x0, x1)) &
             (A[:, 1] >= min(y0, y1)) & (A[:, 1] <= max(y0, y1)))
        A = A[m]
    B = mer[mer.region == a.region] if a.region else mer
    B = B[["center_x", "center_y"]].to_numpy(float)

    print(f"  live     {len(A):,} cells")
    print(f"  MERSCOPE {len(B):,} cells")
    if len(A) < 20 or len(B) < 20:
        raise SystemExit("too few cells to build constellations")

    print(f"\n  building constellations ({a.k_neighbours} neighbours, "
          f"up to {a.max_per_cell} per cell)")
    hA, mA = build_quads(A, a.k_neighbours, a.max_per_cell)
    hB, mB = build_quads(B, a.k_neighbours, a.max_per_cell)
    print(f"    live     {len(hA):,} constellations")
    print(f"    MERSCOPE {len(hB):,} constellations")

    print(f"\n  matching fingerprints (tolerance {a.hash_tol})")
    res = vote(A, B, hA, mA, hB, mB, a.hash_tol,
               (a.scale_min, a.scale_max))
    print(f"    {res['n_quad_matches']:,} fingerprint matches")
    print(f"    largest group agreeing on one transform: "
          f"{res['n_consistent']:,}")

    # ---- the control -------------------------------------------------
    print(f"\n  same test against {a.n_null} scrambled versions")
    rng = np.random.default_rng(0)
    null = []
    lo, hi = B.min(0), B.max(0)
    for i in range(a.n_null):
        Bs = rng.uniform(lo, hi, B.shape)      # same count, same area
        hBs, mBs = build_quads(Bs, a.k_neighbours, a.max_per_cell, seed=i + 1)
        r = vote(A, Bs, hA, mA, hBs, mBs, a.hash_tol,
                 (a.scale_min, a.scale_max))
        null.append(r["n_consistent"])
        print(f"    scrambled {i+1}: {r['n_consistent']:,}")
    nm, ns = float(np.mean(null)), float(np.std(null) + 1e-9)
    z = (res["n_consistent"] - nm) / ns

    print("\n" + "=" * 62)
    print("DOES A PATTERN MATCH?")
    print("=" * 62)
    print(f"  real      {res['n_consistent']:,} constellations agree")
    print(f"  scrambled {nm:.1f} +/- {ns:.1f}")
    print(f"  -> {z:+.1f} sigma above the control")
    if res["best"] is not None and res["n_consistent"] > 0:
        s, ang, tx, ty = res["best"]
        print(f"\n  the transform they agree on: scale {s:.4f}, "
              f"rotation {ang:+.2f} deg, shift ({tx:+.0f}, {ty:+.0f}) um")
    print()
    if z > 5 and res["n_consistent"] >= 10:
        print("  Many independent constellations agree on one transform.")
        print("  That cannot happen by chance -- the cells correspond, and")
        print("  this transform is the one to use.")
    elif z > 3:
        print("  Weak agreement, above the control but not decisively.")
        print("  Worth repeating on another region before relying on it.")
    else:
        print("  No more agreement than scrambled data produces. The local")
        print("  neighbourhood patterns are not shared between the two")
        print("  datasets, so there is no constellation to find.")

    pd.DataFrame(dict(real=[res["n_consistent"]], null_mean=[nm],
                      null_sd=[ns], sigma=[z],
                      quad_matches=[res["n_quad_matches"]])
                 ).to_csv(f"{a.out}/result.csv", index=False)

    if res["votes"] is not None and len(res["votes"]) > 2:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        V = res["votes"]
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].scatter(V[:, 2], V[:, 3], s=6, alpha=.4, c="#1D6F8C")
        ax[0].set_xlabel("shift in x (um)"); ax[0].set_ylabel("shift in y (um)")
        ax[0].set_title(f"what each matching constellation votes for\n"
                        f"a real match is one tight cluster", fontsize=12)
        ax[1].hist(V[:, 1], bins=72, color="#B03A8B")
        ax[1].set_xlabel("rotation voted for (degrees)")
        ax[1].set_ylabel("constellations")
        ax[1].set_title("a real match is one sharp spike", fontsize=12)
        fig.suptitle(f"{res['n_consistent']} agree vs {nm:.0f} scrambled "
                     f"({z:+.1f} sigma)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{a.out}/votes.png", dpi=145)
        print(f"\n  -> {a.out}/votes.png")
    print(f"  -> {a.out}/result.csv")


if __name__ == "__main__":
    main()
