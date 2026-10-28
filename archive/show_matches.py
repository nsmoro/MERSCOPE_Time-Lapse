#!/usr/bin/env python
"""
Draw the matched pairs, labelled by their row in the match table.

So a row of the CSV can be checked against the picture: match #7 is this
live cell, at this position, paired with that MERSCOPE cell. Live cells
are drawn in their registered positions, so a correct pair is two dots
almost on top of each other.

Three views:
  pairs_overview.png  every match across the whole field, each drawn as a
                      line from the live cell to its MERSCOPE partner.
                      Short lines all pointing the same way means a real
                      alignment; long lines pointing everywhere do not.
  pairs_zoom_*.png    a few hundred microns around the most confident
                      matches, numbered by CSV row.
  pairs_table.csv     the match table with the drawn label added, plus
                      both sets of coordinates, so every row can be found.

    python show_matches.py --matches ./final_table/matches_confident.csv \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --reg ./final_table/registration.pkl --out ./pairs
"""
import argparse, os, pickle
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--matches", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--reg", required=True)
ap.add_argument("--out", default="./pairs")
ap.add_argument("--zoom-um", type=float, default=400.0)
ap.add_argument("--n-zooms", type=int, default=3)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

m = pd.read_csv(a.matches)
live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
with open(a.reg, "rb") as f: reg = pickle.load(f)

lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"
W = reg(live[[lx, ly]].to_numpy(float))          # live, registered
M = mer[["center_x", "center_y"]].to_numpy(float)

m = m.reset_index(drop=True)
m.insert(0, "label", np.arange(1, len(m) + 1))
m["live_x"], m["live_y"] = W[m.live_idx, 0], W[m.live_idx, 1]
m["mer_x"],  m["mer_y"]  = M[m.fixed_idx, 0], M[m.fixed_idx, 1]
m.to_csv(f"{a.out}/pairs_table.csv", index=False)
print(f"{len(m)} matches  -> {a.out}/pairs_table.csv")
print(f"  distance: median {m.dist_um.median():.1f} um, "
      f"p90 {m.dist_um.quantile(.9):.1f} um")
dx, dy = (m.mer_x - m.live_x), (m.mer_y - m.live_y)
print(f"  offsets:  dx spread {dx.std():.1f} um, dy spread {dy.std():.1f} um")
print("    (a real alignment has every pair sharing nearly the same offset,")
print("     so both spreads would be small)")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(21, 9))
ax[0].scatter(M[:, 0], M[:, 1], s=1, c="#D9C2D4", linewidths=0, label="MERSCOPE (all)")
ax[0].scatter(W[:, 0], W[:, 1], s=1, c="#BFE0D2", linewidths=0, label="live (all, registered)")
for _, r in m.iterrows():
    ax[0].plot([r.live_x, r.mer_x], [r.live_y, r.mer_y], "-", c="#333", lw=.5, zorder=3)
ax[0].scatter(m.live_x, m.live_y, s=7, c="#1D9E75", zorder=4, label="matched live")
ax[0].scatter(m.mer_x, m.mer_y, s=7, c="#B03A8B", zorder=4, label="matched MERSCOPE")
ax[0].set_aspect("equal"); ax[0].invert_yaxis(); ax[0].legend(markerscale=4, fontsize=9)
ax[0].set_title(f"All {len(m)} matches — each line joins a pair", fontsize=13)
ax[0].set_xlabel("um"); ax[0].set_ylabel("um")

ax[1].scatter(dx, dy, s=9, c="#1D6F8C", alpha=.6)
ax[1].axhline(0, c="#999", lw=.6); ax[1].axvline(0, c="#999", lw=.6)
ax[1].set_aspect("equal")
ax[1].set_xlabel("offset in x (um)"); ax[1].set_ylabel("offset in y (um)")
ax[1].set_title("Offset of every pair\na real alignment is one tight cluster",
                fontsize=13)
fig.tight_layout(); fig.savefig(f"{a.out}/pairs_overview.png", dpi=150); plt.close(fig)
print(f"  -> {a.out}/pairs_overview.png")

top = m.nlargest(a.n_zooms * 40, "confidence")
seeds, used = [], []
for _, r in top.iterrows():
    if all(np.hypot(r.live_x - u[0], r.live_y - u[1]) > a.zoom_um * 2 for u in used):
        seeds.append(r); used.append((r.live_x, r.live_y))
    if len(seeds) >= a.n_zooms: break

for k, s in enumerate(seeds, 1):
    h = a.zoom_um / 2
    fig, ax = plt.subplots(figsize=(11, 11))
    sel = ((np.abs(M[:,0]-s.live_x)<h)&(np.abs(M[:,1]-s.live_y)<h))
    sl  = ((np.abs(W[:,0]-s.live_x)<h)&(np.abs(W[:,1]-s.live_y)<h))
    ax.scatter(M[sel,0], M[sel,1], s=110, c="#B03A8B", alpha=.75, label="MERSCOPE cell")
    ax.scatter(W[sl,0],  W[sl,1],  s=110, c="#1D9E75", alpha=.75, label="live cell (registered)")
    sub = m[(np.abs(m.live_x-s.live_x)<h)&(np.abs(m.live_y-s.live_y)<h)]
    for _, r in sub.iterrows():
        ax.plot([r.live_x, r.mer_x], [r.live_y, r.mer_y], "-", c="#222", lw=1.3, zorder=3)
        ax.annotate(f"#{int(r.label)}", ((r.live_x+r.mer_x)/2, (r.live_y+r.mer_y)/2),
                    fontsize=9, color="#222", weight="bold", zorder=5,
                    ha="center", va="bottom")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.legend(fontsize=10)
    ax.set_title(f"Zoom {k} — {len(sub)} matches here, numbered by row in "
                 f"pairs_table.csv", fontsize=13)
    ax.set_xlabel("um"); ax.set_ylabel("um"); ax.grid(alpha=.2, lw=.5)
    fig.tight_layout(); fig.savefig(f"{a.out}/pairs_zoom_{k}.png", dpi=160); plt.close(fig)
    print(f"  -> {a.out}/pairs_zoom_{k}.png")

print("\nIn each zoom, a correct pair is two dots nearly touching, joined by")
print("a very short line. Long lines crossing each other mean the pairs are")
print("nearby cells rather than the same cell.")
