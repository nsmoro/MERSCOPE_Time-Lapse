# archive

Approaches tried during the investigation that did not work out. Kept
because knowing what was ruled out, and how, is part of the result.

Each script still runs and still has its `--help`.

## Orientation and mapping

| script | what it tested |
|---|---|
| `plot_oriented.py` | the live data drawn under each of the eight orientations |
| `flip_merscope.py` | mirroring or rotating the MERSCOPE side instead |
| `check_orientation.py` | which orientation is consistent with a landmark pair |
| `test_well_mapping.py` | a proposed block-to-block mapping, tested pair by pair |
| `test_marcus_mapping.py` | the field-to-region mapping supplied from the bench |
| `plot_marcus_layout.py` | both datasets coloured by region in that mapping |

The orientation turned out to be a 180 degree rotation, confirmed both from
the bench record and independently from the geometry, where it fits about
thirty times better than the alternatives. The mapping was three
consecutive fields per region. Neither produced cell-level correspondence.

## Transforms that did not help

| script | what it tested |
|---|---|
| `book_fold.py` | folding one region onto its partner like closing a book |
| `fov_reorder.py` | mirroring a block then reordering the fields inside it |
| `plot_mirror_swap_layout.py` | that transform applied across the whole slide |
| `test_deformation.py` | a smooth local deformation, with held-out validation |
| `export_and_tile.py` | exporting in a chosen orientation and tiling |
| `constellation_tiles.py` | constellation matching tile by tile |

The deformation test is worth keeping for its structure: it fits the warp
on half the cells and measures it on the other half, against a shuffled
control. Without that split, a flexible warp fits anything.

## Hand-picked landmarks

| script | what it did |
|---|---|
| `label_cells.py` | number every cell in two patches so pairs can be named by hand |
| `check_dirks_boxes.py` | extract the cells inside hand-drawn boxes and compare |
| `apply_landmark_offset.py` | take an offset from identified pairs, apply it everywhere |
| `overlay_regions.py` | one region overlaid on its partner at matched scale |

These exist because visual judgement and statistical tests disagreed, and
the disagreement needed resolving on the terms of whoever was looking. The
useful part is the spread between hand-picked pairs: pairs that genuinely
correspond all imply the same offset.

## Superseded figures and tables

| script | replaced by |
|---|---|
| `pairs_wide_gate.py` | `pairs_one_to_one.py` |
| `show_matches.py`, `show_matched_pairs.py` | `region_overview_figure.py` |
| `matched_by_region.py` | `region_overview_figure.py` |
| `decode_per_cell.py` | `decode_cells_insitu.py` |

`pairs_wide_gate.py` is still useful for one thing: its `--sweep` mode
shows how the match rate climbs with the search radius while enrichment
stays flat, which is the clearest demonstration that the rate alone is not
a measurement.

`decode_per_cell.py` assigns calls to the vendor's cell table, which needs
a conversion from camera pixels through stage coordinates to mosaic
microns. `decode_cells_insitu.py` avoids all of it by segmenting cells from
the same crop that was decoded.
