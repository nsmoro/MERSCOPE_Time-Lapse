# Live imaging to MERSCOPE: cell matching and MERFISH decoding

Tools for linking single cells tracked in time-lapse microscopy to the same
cells measured by MERSCOPE (Vizgen) spatial transcriptomics, and for
decoding MERFISH barcodes directly from the raw image stacks.

Written for one experiment, but the methods are general: matching two point
sets without a known transform, and deciding honestly whether a match
exists at all.

## The pipeline

Run in this order. Every script has `--help`.

### 1. Match cells

```bash
python pairs_one_to_one.py \
    --live live_coordinates.csv \
    --merscope merscope_cells.csv \
    --flip xy --stat median --method both \
    --out ./matches
```

Pairs each live cell with at most one MERSCOPE cell, and the other way
round. Two methods: `mutual` keeps a pair only when each cell is the
other's nearest, which is strict; `optimal` pairs the whole set at once to
minimise total distance. `--stat median` sets each region's search radius
from its own nearest-neighbour distribution rather than imposing one value.

The output reports, per region, the match rate alongside what the same
procedure gives on scattered cells. That ratio is the result; the rate on
its own is not, because it can be set to almost any value by widening the
radius.

`pairs_adaptive_gate.py` does the same without the one-to-one constraint,
if you want to see how many pairs are contested.

### 2. Look at what matched

```bash
python region_overview_figure.py --pairs ./matches/pairs_optimal_median.csv \
    --live live_coordinates.csv --merscope merscope_cells.csv \
    --tiles 3 --out ./figures

python tiles_plain.py --live live_coordinates.csv \
    --merscope merscope_cells.csv --all --tiles 2 --out ./plain
```

The first draws one figure per region: a locator strip showing where on the
slide you are, then tiles with each matched pair numbered the same on both
sides. The second draws every cell with nothing highlighted, for judging
the patterns by eye rather than by algorithm.

### 3. Summarise

```bash
python match_summary_xlsx.py --pairs ./matches/pairs_optimal_median.csv \
    --live live_coordinates.csv --merscope merscope_cells.csv \
    --gate-um 52 --out summary.xlsx
```

Per-region counts on both sides, percentages, and the chance and enrichment
columns.

### 4. Decode transcripts from the raw images

```bash
python decode_fov.py --raw /path/to/merfish_raw_data --fov 0 \
    --crop 1024 --out ./decode

python decode_cells_insitu.py --raw /path/to/merfish_raw_data --fov 0 \
    --crop 1024 --out ./decode_cells
```

Implements MERFISH decoding directly, so every threshold is adjustable.
Sweeps both the intensity cutoff and the barcode-matching distance, and
reports gene calls against blank calls at each setting. The second version
segments cells from the same crop and gives per-cell distributions.

### Checking the transform independently

```bash
python constellation_match.py --live live_coordinates.csv \
    --merscope merscope_cells.csv --region R18 --positions 22,23,24 \
    --out ./constellation
```

Matches cells by the shape their neighbours make rather than by position.
Needs no transform, no orientation and no mapping — so it is the strongest
check that an alignment has not simply been missed.

## The ideas worth knowing

**The search radius determines the match rate.** Cells sit some distance
apart; search further than that and every cell finds a partner whether or
not it has one. A match rate quoted without its radius means nothing.

**Enrichment over chance is the number that cannot be tuned.** Run the
identical procedure against scattered cells at the same density, and
divide. Widening the radius raises both, so the ratio stays honest.

**Blank barcodes measure the decoding error rate directly.** A MERFISH
codebook contains patterns assigned to no gene. Nothing in the sample can
produce one, so every blank called is a known mistake. Comparing the
gene-to-blank ratio against what the codebook composition gives by chance
turns "is this threshold right?" into a measurement.

**Validate before trusting a negative.** Every script here was first
checked against simulated data with a known answer: the matching reaches
96–97% precision, the constellation method recovers a 12° rotation and a
5 mm shift through 30% cell loss and 60% contamination, and the decoder
recovers 297 of 300 planted transcripts with no false positives. A tool
that cannot find a match that exists tells you nothing when it finds none.

## Input format

CSVs with coordinate columns. These names are detected automatically:
`x_um`/`y_um`, `center_x`/`center_y`, `CentroidXglobal`/`CentroidYglobal`,
`x`/`y`.

Live coordinates additionally need a `Position` column giving the field of
view. MERSCOPE coordinates need a `region` column.

Decoding reads a MERSCOPE raw-data folder containing `codebook_*.csv`,
`dataorganization.csv`, and `data-compressed/` holding
`stack_<round>_<fov>.jp2.zip`.

## Dependencies

```
numpy pandas scipy matplotlib scikit-image
imageio imagecodecs      # reading JPEG2000 image stacks
openpyxl                 # spreadsheet output
cellpose                 # optional, for segmentation
```

## archive/

Approaches tried along the way that did not work out, kept because the
negative results are part of the record: alternative orientation and
mapping tests, a non-rigid deformation fit with held-out validation,
per-field reordering, hand-picked landmark tools, and several figure
variants. Each still runs and still has its `--help`.

## Licence

MIT. See `LICENSE`.
