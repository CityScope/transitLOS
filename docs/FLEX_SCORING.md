# Flexible, recomputable scoring (`transitlos.flex`)

The original pipeline scores a stop once and bakes the result into the street isochrones. `transitlos.flex` splits the work so that **the scoring function can be changed after processing** and recomputed in milliseconds, on the server or in the browser.

```
GTFS ──► station profiles (≤ P per parent station)   ┐   independent of the scoring function
street network + stations ──► stations per H3 cell (≤ K)  ┘   (monotone scoring assumed, see below)
profiles + cells + ScoreParams ──► stop scores ──► cell scores ──► roll-ups      ← the cheap, repeatable step
```

## What is stored
| File | Content |
|---|---|
| `stop_profiles.parquet` | one row per parent station: `lat, lon, cropped` and for each of P slots `mode_j` (0 bus, 1 tram, 2 rail; -1 empty), `h_j` headway (min), `v_j` speed (km/h), `cut_j` minimum-speed cutoff of the candidate |
| `cell_stops.parquet` | one row per served H3 res-10 cell: `h3`, K station indexes `stop_k` and distances `dist_k` (m, uint16), `flag`, `cropped` |
| `scores_default.parquet` | the cell scores under the default parameters (0-100) and the winning station slot / profile |
| `params.json` | the `ScoreParams` and `Config` used |
| browser export | `stop_*.bin`, `cell_*.bin`, `meta.json`, `score.js`, `viewer.html` (`transitlos.flex.export`) |

`cropped` counts only items **genuinely cut by the cap** (non-dominated, above the floors, over the limit). Items removed by the floors or because another is objectively better are not counted. `flag`: bit0 the cap was needed, bit1 the candidate-list limit was hit, bit2 the search refused labels at a node touching the cell.

## Station profiles (`profiles.py`)
Same candidates as `transitlos.scoring.direction_subset` (3 mode tiers × minimum trip speed in 5 km/h steps; headway = the better direction of the merged departures; speed = Σdistance/Σtime), but all of them are kept. Then, per station: **exact dominance** (another candidate has ≥ mode, ≤ headway, ≥ speed) → **tolerance merge** (best first, 10 % headway / 5 % speed) → **cap** to P (below), which keeps a diverse set that prefers good values. Optional **floors** (`h_max`, `v_min`, `s_min`, `p_min`) exist but are OFF by default: nothing is dropped for being slow or infrequent unless you ask for it.

## Stations per cell (`cells.py`, `core.multilabel_search`, `core.process_cells`)
1. Stations (parent stations, 150 m grouping) are snapped to the nearest street edge and become two sources (the edge ends, offset by snap distance and position).
2. **Multi-label Dijkstra** (C): every node keeps up to `Kc` labels, one per station. Labels arrive in non-decreasing distance, so a new label only needs a quality comparison with the labels already at the node; a label whose station is dominated by a closer one (every profile matched, within tolerance) is dropped and **not propagated** (exact for monotone scoring). Overlapping paths are fine: each station settles each node once at its own shortest distance.
3. Distance of a cell = distance **until an edge touches the cell** (min over the edges touching it of `min(d_u + a, d_v + L − b)`, `a`/`b` = where the edge enters/leaves the cell) **+ the cell radius** (66 m for res 10), so a barely connected cell is not credited with a well connected neighbour.
4. Per cell: dominance in distance order → cap to K (optional floor `p_min`, off by default).

## The cap (`select_core` in `flexcore.c`)
Used only when there are more candidates than the cap. Used for profiles (axes: headway, mode, speed) and for cell stations (distance, headway, mode, speed). Order of selection, stopping at the cap:
1. **equilibrium** – the best default score; for cells also the **nearest station, kept exactly** (so "is there any transit within X m of people" stays answerable whatever the scoring: `dist_0` of a cell is its true nearest station within the search limit);
2. **tolerant extremes** – for every subset of 1, then 2, 3 … axes (axis order: distance, headway, mode, speed): the candidate with the best default score among those within `tau_band` (0.3) of the best *normalised* geometric-mean value of that subset; subsets containing *mode* run once per mode level (rail, tram, bus);
3. **diversity fill** – repeatedly add the candidate maximising `(quality/best)^γ × distance-to-the-chosen-set` (γ 2) in the normalised axis space; stop when it is within `delta` (0.05) of something chosen (no near-copies).

Measured on Manhattan (8,422 cells) against an uncapped reference, under 140 scoring functions never used for the selection (including each axis switched off completely): **P = 10, K = 15 default (chosen after a sweep; with floors: P = 10, K = 10 gave mean error 0.005 points, 0.2 % of cells off by more than 1 point; without floors K = 10 gave 0.017, K = 12 0.010, K = 14 0.008, K = 16 0.006)**; mode never lost. Smaller caps and the full tables: `usa_flex/analysis/` and the study notes.

## Changing the scoring (`scoring.py`, `shapes.py`)
`ScoreParams` holds everything; `with_()` changes fields.

* **Parameters** – `mrc` (value of bus/tram/rail), speed `v_anchor/v_bonus/v_n`, headway `h_sat/h_p/h_q`, walk `walk_speed/plateau_m/t0_base/k/p_shape`, axis weights `w_mode/w_speed/w_headway` (**0 switches that axis off**).
* **Shapes** – `headway_curve`, `speed_curve`, `walk_curve` are `Curve`s: `Curve()` is the original analytic shape; any other shape is a sampled table (`shapes.py`: exponential, power, logistic, steps, linear, gaussian, hard buffer, `piecewise([...])`, or `Curve.from_function(f, lo, hi)`). The walk curve is `D0(u)` with `u = (t − plateau)/t0(stop score, mode)`; `k = 0` removes the coupling.
* **Aggregation** – `agg="nested"` (default `min(1, (MRC^wm·V^ws)^(1/(wm+ws))·H^wh)` = `sqrt(MRC·V)·H`) or `"power_mean"` with exponent `agg_p` (1 arithmetic, 0 geometric, −1 harmonic) – the legacy `harmonic`/`geometric` modes of `transitlos.scoring` are reproduced exactly.

```python
from transitlos.flex import ScoreParams, shapes, core
p = ScoreParams(headway_curve=shapes.headway_exponential(sat=5, scale=40), walk_curve=shapes.walk_gaussian(), w_mode=0.0)
los, slot, profile = core.recompute_cells_fast(cell_station, cell_dist, prof, p)    # 10 M cells in ~0.2-0.4 s
```
The same `p.to_json()` drives `frontend/score.js` in the browser (`FlexScore.prepare / stopScores / cellScores`); `export.shape_catalog()` lists the presets for a UI.

**Assumption.** Dominance and the cap are exact only if the scoring is monotone: better mode, higher speed, lower headway and shorter distance never lower a score. Any curves that respect this are safe; a curve that rewards *worse* service would not be (the cap could have removed the winner).

## Catchments: who does each stop serve? (`catchment.py`)
A stop **wins** a cell when it is the station that gives the cell its level of service (the station of the winning slot of `core.recompute_cells*`). For every stop the map sums any additive census column (population, jobs, age groups, ...) over the cells it wins; the stop size shows that number and a click prints it. Because the winner depends on the scoring, **the catchments are recomputed with the scores** (browser and Python identical).

* `block_counts_to_cells(blocks, cells, columns)` spreads the counts of a polygon layer (census blocks: population, LODES jobs, under 18, 65+) over the served cells assuming uniform density inside each polygon: polygons larger than two cells give each cell whose centre they contain `cell_area / polygon_area` of their count; smaller polygons give everything to the cell of their representative point. Counts in cells no stop reaches are attributed to no stop.
* `winner_station`, `stop_catchment` (sum per stop), `cells_won`.

## The map (`frontend/`, built by `export.export_for_browser`)
`viewer.html` + `score.js` (scoring), `editor.js` (scoring editors), `glmap.js` (WebGL2 map, no dependencies). A state is a folder of binary tables (`stop_*.bin`, `cell_*.bin`, `cell_counts.bin`, `cell_geom.bin`, `meta.json`); `web/index.json` lists the states for the selector. Serve the folder and open `viewer.html` (`?state=Vermont&data=path`).

* **Scoring editors** (left panel): for the headway, speed and walk curves choose the shape family (default analytic, exponential, power, logistic, steps, linear, gaussian, hard buffer, **custom points** you drag on the plot) and its parameters; the plot shows the curve against the default (dashed). Also: aggregation (nested, arithmetic / geometric / harmonic mean, near-minimum), the three axis weights (0 = ignore), the value of each mode, walking speed, plateau, the walk scale per mode and the coupling k. **Save / load**: the whole scoring is one JSON (`ScoreParams.to_json`), copy it, download it, paste another.
* **Map**: cells coloured by level of service or by the change versus the default scoring; **stop size by** any census count (legend with example sizes); click a stop for its catchment numbers, its cells won and its service patterns with their scores; click a cell for its score, nearest stop, census counts and stations.
* **Statistics** (live): people in cells a stop reaches, mean score of people, people within X m (editable) of a stop, people with a score >= 50.
* Tested in headless Chromium against numpy/C (`tests/test_flex_viewer.py`, and `usa_flex/analysis/viewer_test.py` on real states): scores agree to 1e-5 points and catchments exactly under every edit.
* Zoomed out (a cell is under 4 px) cells are drawn as points; zoomed in, as hexagons.

## Speed (Manhattan: 8,422 cells, 983 stations; 20 threads)
Search 0.55 s, cell mapping 0.2–0.8 s, cap 0.01 s, recompute 4 ms; recompute throughput 28 M cells/s in C (40× numpy), 4 ms in the browser for the whole area; file size 35 B per cell, 71 B per station. JS and C agree to 7·10⁻⁶ points for 44 scoring definitions (`usa_flex/analysis/browser_test.py`).

## Code map
`flex/scoring.py` (params, curves, numpy) · `flex/shapes.py` · `flex/flexcore.c` + `flex/core.py` (ctypes; compiled on first import, needs `gcc`, OpenMP optional; falls back to `~/.cache/transitlos` if the package folder is read-only) · `flex/profiles.py` · `flex/cells.py` · `flex/catchment.py` · `flex/export.py` + `flex/frontend/` (`viewer.html`, `score.js`, `editor.js`, `glmap.js`) · street graph helpers in `UrbanAccessAnalyzer.csr_network` (`pip install h3ronpy`) · tests `tests/test_flex_*.py` (scoring, cap, network search, catchment, viewer) · synthetic cap study `tools/flex_cap_synthetic.py`.

## Device: GPU when CUDA is usable, CPU otherwise

`transitlos.flex.backend` decides where the array work runs: `device()` is `'cuda'` when CuPy imports and a CUDA device answers, else `'cpu'`; `TRANSITLOS_DEVICE=cpu|cuda|auto`
forces it (default auto). A missing CuPy, a missing driver or a failing device always mean the CPU, never an error. `transitlos.flex.xp_scoring.XPData` scores a whole
sample as array code (the definition of `scoring.py` / `flexcore.c` / `score.js`) on whichever device that is; the CPU path of the normal engine stays the C core.
