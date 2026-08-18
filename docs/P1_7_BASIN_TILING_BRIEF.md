# P1.7 — Basin domains + HUC12-tiled parallel solver (HPC brief)

User directive 2026-08-18: **the 2-D domain is the basin, never a rectangle.**
The hydrodynamic model simulates the *whole* catchment contributing to the
trigger gauge; when that is too big for one device, the **hydrologic units
(HUC12) are the decomposition** across devices/nodes.

## What changed (fork v2, commits 678cbf17 + this one)

| piece | where | what |
|---|---|---|
| basin domain | CREST-AI `hf_data/hucdomain.py` (runner Space) | gauge → HUC12 (point-in-polygon) → `tohuc` chain upstream = the basin (reproduces USGS drainage areas: Spoon R 4,240 vs 4,241 km²). Writes `domain_huc.tif` (int16 unit labels, −1 outside, on the 3DEP lattice at the spec's `dem_res`) + `domain.geojson` into the EF5 dir → both ride in the forcing tar. Spec carries `domain{unit,n_hucs,area_km2,bbox,n_active,n_bbox,...}` and `huc12s[]`; `bbox_basin` is now the union's raster extent. WBD tables: dataset `vincewin/CREST_data` `wbd/huc12/` (index + per-HUC4 geometry). |
| solver mask | `crestimap/solver.py` | `SWESolver(active=mask)`: cells outside the basin are sinks (state zeroed every step) — free outflow at the pour point, nothing crosses a divide. `mask_state()` also applied after the channel-stage nudge and to the pre-wet/resume state. |
| plumbing | `crestimap/domain.py`, `event.py`, `session.py` | `EventConfig.domain_path` (default `<ef5_output_dir>/domain_huc.tif`), `require_domain=True` → **no raster = error, never a rectangular run**. Manifest gets `domain{n_units,n_active,n_bbox,active_frac,geojson}`; `domain.geojson` is copied into every visit's out dir so it publishes. |
| worker | `crestimap/worker.py` | skips specs without `domain` (pre-basin bundles await the runner's re-enqueue — do NOT solve them as rectangles); `expected_cells(spec, n_devices)`; manifest gets `huc12s` + merged `domain`; `--devices cuda:0,cuda:1,...` / `--tiles N`. |
| **tiled solver** | `crestimap/tiled.py` | `TiledSolver(z, labels, dx, dy, devices, n_tiles, halo=2)`: units → contiguous groups balanced by cell count (along the domain's long axis); tile = bounds of owned cells + halo; `active = owned ∪ ghost` (ghost = other tiles' basin cells within Chebyshev distance ≤ halo); SSP-RK2 with a halo `index_copy` after **each** stage; global dt = min over tiles. **Owned cells are bit-identical to the monolithic masked solve** (`tests/test_tiled.py` T1: orders 1/2, 1/3/5 tiles, max|Δ| = 0). Ghost cells ≈ 4 % of owned at realistic scale; tile rects ≈ 1.3× owned (a bounding box is ≈ 2×). |
| session | `session.py` `chunk()` | when `cfg.devices` has > 1 entry (or `n_tiles > 1`) and labels exist: scatter → `TiledSolver.run` → gather; frames gather lazily only when due; maxdepth kept per tile. Junction/rollback semantics unchanged. |

## What to run

```bash
# same as before, plus the device list; --max-cells is now PER DEVICE
python -m crestimap.worker --sessions --publish --device cuda:0 \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 --max-cells 100000000 \
    --repo vincewin/CREST_data --crest-ai $SCRATCH/crest_worker/space_ref
```

Re-sync `space_ref` (hf_data now has `hucdomain.py`, `eventsim.py` with the
basin domain, `eventstore.py` with `domain` in the index summary) and
`git pull` the fork (≥ 2e210d2b) **at the job boundary — before claiming
another job**: an OLD worker on a NEW spec would solve the union's bounding
rectangle (it ignores `domain_huc.tif`), which is exactly what must never
be published again. Old queued bundles (no `domain` in the spec) are
skipped by the new worker. The runner has re-enqueued every listed event
with a basin domain and a `cold` stamp (spec `cold: <ISO>`): the worker
drops any resident session for that episode and re-solves the whole
anchored span from the channel pre-wet, so no box-era frame survives.

## Validation asked of the HPC

1. `CRESTIMAP_TEST_DEVICE=cuda python crestimap/tests/test_tiled.py` and
   `CRESTIMAP_TEST_DEVICE=cuda:0,cuda:1 python crestimap/tests/test_tiled.py`
   (multi-GPU: tolerance 1e-5 — different devices may differ in reduction
   order for the CFL max; report the max|Δ|).
2. One real bundle (e.g. Spoon River 05570000, 56 units, 5.9 M active /
   13.4 M raster cells at 30 m): wall time and peak memory for 1 GPU
   (monolithic) vs 2 and 4 GPUs (tiled). Expected: tile rects sum ≈ 1.3–1.6×
   active cells, so 4 GPUs each hold ≈ 2.4 M cells instead of 13.4 M, and
   step time scales roughly with the largest tile.
3. Report the ghost-cell count and its fraction of owned (logged by
   `TiledSolver.__init__`) — the exchange cost is that many `index_copy`
   elements × 3 fields × 2 stages per step.
4. If multi-node is wanted: the tile/halo tables in `TiledSolver` are the
   contract; wrap `_exchange` in `torch.distributed` sends of the same
   `(idx_here, idx_there)` pairs. Do not change the per-tile physics.

## Known follow-ups (Space side, not yours)

* EF5's own hydrologic domain (`pipeline.basin_bbox`, a √area square centred
  on the gauge) **clips elongated basins** (Spoon 3,884 of 4,241 km²; Pope Ck
  199 of 449). The 2-D domain no longer depends on it, but runoff forcing in
  the clipped headwaters is zero until EF5's domain is also derived from the
  HUC12 union bounds. Being scheduled with the fleet-state implications.
