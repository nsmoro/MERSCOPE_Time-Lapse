#!/usr/bin/env python
"""
Per-region match summary as a formatted spreadsheet.

Counts both sides separately, because they answer different questions:
how much of the time-lapse found a partner, and how much of MERSCOPE was
accounted for. Those differ when several live cells fall nearest the same
MERSCOPE cell, which happens at wide search radii.

A chance column is included. At a search radius comparable to the spacing
between cells, a sizeable fraction of cells will have SOMETHING within
reach whether or not any real correspondence exists, so the raw percentage
on its own overstates what has been found.

    python match_summary_xlsx.py \
        --pairs ./pairs_swapped/pairs_gate52um.csv \
        --live ./global_rebuilt/global_xy_last_tp.csv \
        --merscope merscope_all_regions.csv \
        --gate-um 52 --out match_summary.xlsx
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--live", required=True)
ap.add_argument("--merscope", required=True)
ap.add_argument("--gate-um", type=float, required=True)
ap.add_argument("--region-order",
                default="R9,R11,R12,R13,R14,R16,R17,R18,R20")
ap.add_argument("--per-region", type=int, default=3)
ap.add_argument("--out", default="match_summary.xlsx")
a = ap.parse_args()

P = pd.read_csv(a.pairs, comment="#")
live = pd.read_csv(a.live); mer = pd.read_csv(a.merscope)
lx = "x_um" if "x_um" in live else "CentroidXglobal"
ly = "y_um" if "y_um" in live else "CentroidYglobal"

regions = [r.strip() for r in a.region_order.split(",")]
positions = sorted(live.Position.unique())
groups = {}
for k, reg in enumerate(regions):
    lo = k * a.per_region
    hi = (k + 1) * a.per_region if k < len(regions) - 1 else len(positions)
    groups[reg] = positions[lo:hi]

rng = np.random.default_rng(0)
rows = []
for reg in regions:
    ps = groups[reg]
    n_live = int(live.Position.isin(ps).sum())
    gm = mer[mer.region == reg]
    n_mer = len(gm)
    sub = P[P.region == reg] if "region" in P else P.iloc[0:0]
    m_live = int(sub.live_row.nunique()) if "live_row" in sub else len(sub)
    m_mer = int(sub.fixed_id.nunique()) if "fixed_id" in sub else len(sub)

    chance = np.nan
    if n_mer > 20 and n_live > 20:
        B = gm[["center_x", "center_y"]].to_numpy(float)
        t = cKDTree(B); lo_, hi_ = B.min(0), B.max(0)
        chance = float(np.mean([
            (t.query(rng.uniform(lo_, hi_, (n_live, 2)), k=1)[0]
             < a.gate_um).mean() for _ in range(5)]))

    rows.append(dict(
        Region=reg,
        Positions=f"{ps[0]}-{ps[-1]}",
        Cells_timelapse=n_live,
        Cells_MERSCOPE=n_mer,
        Matched_timelapse=m_live,
        Matched_MERSCOPE=m_mer,
        Pct_timelapse=m_live / n_live if n_live else np.nan,
        Pct_MERSCOPE=m_mer / n_mer if n_mer else np.nan,
        Pct_expected_by_chance=chance,
        Enrichment=(m_live / n_live / chance) if (n_live and chance) else np.nan,
        Median_separation_um=float(sub.dist_um.median())
        if len(sub) and "dist_um" in sub else np.nan))

T = pd.DataFrame(rows)
tot = dict(Region="TOTAL", Positions=f"{positions[0]}-{positions[-1]}",
           Cells_timelapse=T.Cells_timelapse.sum(),
           Cells_MERSCOPE=T.Cells_MERSCOPE.sum(),
           Matched_timelapse=T.Matched_timelapse.sum(),
           Matched_MERSCOPE=T.Matched_MERSCOPE.sum())
tot["Pct_timelapse"] = tot["Matched_timelapse"] / tot["Cells_timelapse"]
tot["Pct_MERSCOPE"] = tot["Matched_MERSCOPE"] / tot["Cells_MERSCOPE"]
tot["Pct_expected_by_chance"] = float(np.average(
    T.Pct_expected_by_chance, weights=T.Cells_timelapse))
tot["Enrichment"] = tot["Pct_timelapse"] / tot["Pct_expected_by_chance"]
tot["Median_separation_um"] = float(P.dist_um.median()) if "dist_um" in P else np.nan
T = pd.concat([T, pd.DataFrame([tot])], ignore_index=True)

print(T.to_string(index=False))

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Match summary"
HDR = ["Region", "Positions", "Cells in time-lapse", "Cells in MERSCOPE",
       "Matched (time-lapse)", "Matched (MERSCOPE)", "% of time-lapse",
       "% of MERSCOPE", "% expected by chance", "Enrichment",
       "Median separation (um)"]
ws.append(HDR)
for _, r in T.iterrows():
    ws.append([r.Region, r.Positions, r.Cells_timelapse, r.Cells_MERSCOPE,
               r.Matched_timelapse, r.Matched_MERSCOPE, r.Pct_timelapse,
               r.Pct_MERSCOPE, r.Pct_expected_by_chance, r.Enrichment,
               r.Median_separation_um])

thin = Side(style="thin", color="D0D0D0")
bd = Border(left=thin, right=thin, top=thin, bottom=thin)
for c in range(1, len(HDR) + 1):
    cell = ws.cell(1, c)
    cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor="1F3A5F")
    cell.alignment = Alignment(horizontal="center", vertical="center",
                               wrap_text=True)
    cell.border = bd
ws.row_dimensions[1].height = 42

last = ws.max_row
for row in range(2, last + 1):
    is_tot = ws.cell(row, 1).value == "TOTAL"
    for c in range(1, len(HDR) + 1):
        cell = ws.cell(row, c)
        cell.font = Font(name="Arial", size=10, bold=is_tot)
        cell.border = bd
        cell.alignment = Alignment(horizontal="center")
        if is_tot:
            cell.fill = PatternFill("solid", fgColor="E8EDF2")
        if c in (3, 4, 5, 6):
            cell.number_format = "#,##0"
        if c in (7, 8, 9):
            cell.number_format = "0.0%"
        if c == 10:
            cell.number_format = "0.00"
        if c == 11:
            cell.number_format = "0.0"

widths = [9, 11, 17, 17, 18, 18, 14, 14, 18, 12, 18]
for i, w in enumerate(widths, start=1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.freeze_panes = "A2"

n = last + 3
ws.cell(n, 1, "Notes").font = Font(name="Arial", size=11, bold=True)
notes = [
    f"Search radius {a.gate_um:.0f} um. A pair is a live cell and the "
    f"nearest MERSCOPE cell within that radius, after aligning each region.",
    "Matched (time-lapse) counts distinct live cells; Matched (MERSCOPE) "
    "counts distinct MERSCOPE cells. They differ where several live cells "
    "fall nearest the same MERSCOPE cell.",
    "% expected by chance is what the same number of cells would give "
    "scattered at random over the same area, at the same radius.",
    "Enrichment is % of time-lapse divided by % expected by chance. "
    "Near 1.0 means the pairs reflect how close cells are, not which cell "
    "is which.",
]
for k, t in enumerate(notes):
    c = ws.cell(n + 1 + k, 1, t)
    c.font = Font(name="Arial", size=9.5)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=n + 1 + k, start_column=1,
                   end_row=n + 1 + k, end_column=11)
    ws.row_dimensions[n + 1 + k].height = 28

wb.save(a.out)
print(f"\n  -> {a.out}")
