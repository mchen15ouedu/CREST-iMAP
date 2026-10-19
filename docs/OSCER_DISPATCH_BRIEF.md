# OSCER — always-on dispatcher + 48 h GPU jobs for the 2-D event queue

**For:** the Claude Code session on OSCER (OU's cluster). You are building the
replacement for the released HPC GPU worker (server2017, retired 2026-10-16).
Same ground rules as `P1_WORKER_BRIEF.md` / `P1_6_MULTI_EVENT_BRIEF.md`:

- Push only to this fork's `v2` branch (`github.com/mchen15ouedu/CREST-iMAP`).
  Never push to CREST_AI. Never change any Hugging Face Space, Space variable or
  secret. Never write to the `vincewin/CREST_data` dataset except through the
  existing queue protocol and `eventstore.publish_event` (below).
- Never print, log or commit the HF token. Keep it in a `chmod 600` file.
- **Basins, never rectangles; resolution is never coarsened.** Solve exactly
  the domain in the job spec (`domain_huc.tif` in the bundle) at the spec's
  `dem_res`. A job too big for the GPU is left unclaimed, never shrunk or
  coarsened by the worker.
- Never touch `events/pinned.json` and never delete or overwrite event folders
  by hand. Pinned events are the user's demo set.

## The situation

| Piece | Where | Status |
|---|---|---|
| Event detection, EF5 hydrology, bundle enqueue | HF Space `vincewin/CREST_runner` (hourly tick) | running, unchanged |
| Job queue | public dataset `vincewin/CREST_data`, `events/queue/` | unchanged protocol |
| ZeroGPU worker | HF Space `vincewin/CREST_gpu_worker` | running; ~30 GPU-min/day, jobs ≤ 16 M bbox cells |
| **OSCER always-on CPU node (no GPU)** | yours | **dispatcher — build** |
| **OSCER GPU partitions (≤ 48 h wall)** | yours | **GPU worker jobs — build** |

The user has OSCER nodes with no time limit but **no GPU**, and GPU partitions
with a **48 h wall limit**. So: a long-running **dispatcher** on the CPU node
watches the queue and submits **GPU jobs** with `sbatch`. Each GPU job runs the
existing session worker, drains the queue, and exits when idle or before the
wall. Resident sessions are **checkpointed to shared disk**, so a job ending
at 48 h costs a few minutes, not a re-solve.

Measured load to size things (HPC era, all 27,488 queue commits):
- ~300 GPU-h/month typical, ~650 in a busy flood month, on an RTX 4500 Ada
  (6.7e7 cell-steps/s). An H100 is ~4–5× faster for this solver.
- Cost model: GPU-min ≈ 1.0–1.45 × (M bbox cells) × (simulated hours) on the
  RTX 4500 Ada. Median job 2.3 M bbox cells; 90th pct 15 M; max 120 M.
- Solver memory ≈ 300–350 bytes per bbox cell + ~1 GB. A 120 M-cell basin
  fits one 80 GB card (≈ 37–43 GB).
- ~40 new events/month; active episodes are re-enqueued every hour (each
  re-enqueue REPLACES the spec under the same id).

## Step 0 — environment probes (do these first, report results)

1. **GPU partitions you can use:** `sinfo -s`, `sacctmgr show assoc user=$USER`.
   Note partition names, GPU types, max wall, and per-user job limits. OSCER
   publicly lists `sooner_gpu_test`, `..._h100`, `..._h100_dual`,
   `..._h100_quad`, `..._ada` (RTX 6000 Ada 50 GB) — verify, don't assume.
2. **Network from a GPU node — this decides the design:**
   ```bash
   srun -p <gpu partition> --gres=gpu:1 -t 10 bash -c \
     'curl -sS -o /dev/null -w "%{http_code}\n" https://huggingface.co/api/datasets/vincewin/CREST_data; \
      curl -sS -o /dev/null -w "%{http_code}\n" https://prd-tnm.s3.amazonaws.com/; nvidia-smi -L'
   ```
   Both 200/403-style answers (any HTTP code) mean outbound HTTPS works →
   **Mode A**. A timeout / "could not resolve" means no internet on GPU nodes →
   **Mode B**. Also run the same curl on the CPU dispatcher node (it must work
   there).
3. **Python env** with torch+CUDA, numpy, rasterio, shapely, huggingface_hub,
   pyarrow, requests (conda/mamba on `$SCRATCH` or project space). Run the
   solver test-suite on a GPU node:
   `CRESTIMAP_TEST_DEVICE=cuda python -m pytest crestimap/tests -q`
   (all must pass; `test_session.py` is the slow one).
4. **Storage:** pick a shared directory visible from the CPU node and the GPU
   nodes for checkpoints, DEM cache, HF cache and job dirs; check its purge
   policy (checkpoints must survive ≥ 7 days).
5. **HF write probe:** with the token, upload then delete a tiny file
   `oscer_probe/<date>.txt` in `vincewin/CREST_data` (one commit each). Stop
   and report if it returns 401/403 (token lacks write scope).

## The code you depend on (read before building)

- `crestimap/worker.py` — the queue worker. `--sessions` = P1.6 resident-
  session scheduler (`run_sessions`, `process_visit`); `--publish`;
  `--max-cells`; `--devices/--tiles` (multi-GPU tiling); `expected_cells()`;
  claim/heartbeat (`claim`, `_start_heartbeat`, 240 s), stale = 600 s;
  `_spec_unchanged` re-check before publish; `_cleanup_queue` consumes the
  entry only if the spec's `queued` stamp is unchanged; `_report_failure`
  writes `<id>.failed.json`.
- `crestimap/session.py` — `EventSession(cfg, resume=True)`, `begin_visit`,
  `chunk(sim_s, device)`, `end_visit`, `state_tuple()` / `set_state()` (the
  state is already designed to be picklable — the ZeroGPU worker pickles it).
- `run_worker.sh` — the old nohup launcher (env vars: `CREST_AI_DIR`,
  `WORKER_WORK_ROOT`, `EVENT_DEM_CACHE`, `EVENT_MAX_CELLS_GPU`, `EVENT_SESSIONS`).
- CREST_AI `hf_data/` (`eventsim._update_archive`, `eventsim._make_pngs`,
  `eventstore.publish_event`) is imported from a CREST_AI checkout
  (`--crest-ai` / `CREST_AI_DIR`). **Use a fresh clone of
  `https://github.com/mchen15ouedu/CREST_AI.git` `main` and `git pull` it before
  every GPU job** — GitHub `main` is now kept identical to the deployed
  Spaces (the old `space_ref` sync is obsolete). `main` ≥ `731ab17` has the
  retention rules that protect the pinned demo events; the worker must refuse
  to publish if `hf_data/eventstore.py` lacks `def pinned_ids` (old code would
  strip demo events).
- The queue protocol, unchanged:
  ```
  events/queue/<id>.json            spec (domain, bbox_basin, t0, t_end, sim_start, dem_res, cold, queued, …)
  events/queue/<id>.forcing.tar.gz  EF5 grids + domain_huc.tif + domain.geojson (+ init_depth_*.tif)
  events/queue/<id>.claim           {"worker": ident, "hb": "%Y-%m-%dT%H:%M:%SZ", "queued": spec.queued}
  events/queue/<id>.failed.json     error report (job consumed)
  ```
  Claims are exclusive; the ZeroGPU worker skips any fresh claim with another
  `worker` ident and vice versa. Both run at once safely.

## What to build

### 1. Worker: wall-clock aware, checkpointed (`crestimap/worker.py`, sessions mode)

Add, without changing default behaviour (flags off = today's daemon):

- `--max-hours H` — hard deadline = start + H. Stop **claiming** when the
  estimated cost of the next job does not fit the remaining time (estimate with
  the cost model above, scaled by a measured speed factor for this GPU — record
  it in the validation report; be conservative). At deadline − margin, finish
  the current **chunk**, checkpoint, release claims, exit 0.
- **SLURM warning:** handle `SIGUSR1` (sbatch `--signal=B:USR1@1800`) and
  `SIGTERM` the same way as the deadline (set a flag; act at the next chunk
  boundary; never kill mid-write).
- **Chunked visits:** drive each visit with `session.chunk(sim_s)` in slices of
  ~10–20 wall-min (adapt `sim_s` from the measured rate, like the ZeroGPU
  worker does), so a stop request is honoured within one chunk and a visit can
  continue in the next job.
- `--checkpoint-dir DIR` — at every clean stop and every ~30 min:
  for each resident session write `DIR/<event_id>/session.pt` (state tensors
  moved to CPU, via `torch.save` of the session or of `state_tuple()` plus the
  metadata needed to rebuild it: cfg, bbox, cold stamp, visits, the spec's
  `queued` stamp, mid-visit position) + `meta.json`; write to a temp name and
  `os.replace` (atomic). Also persist the processed `(event_id, queued)` set.
  On start, **restore** sessions from `DIR`; drop a checkpoint whose
  `bbox_basin`, `cold` stamp or domain differs from the current spec (then the
  normal `init_depth` resume path applies). Delete checkpoints of events that
  ended (index `status == "ended"`) or are older than 7 days.
- `--idle-exit-min M` — exit (after checkpointing) when nothing has been
  claimable for M minutes.
- **On every exit:** delete this worker's own `.claim` files (a released claim
  lets the next job or ZeroGPU take the event at once instead of after the
  600 s stale timeout). Never delete specs/bundles except through the existing
  `_cleanup_queue` / `_report_failure`.
- **Ident / provenance:** `worker` = `oscer:<hostname>:<SLURM_JOB_ID>`; set
  `EVENT_ENGINE_IDENT` to the same so manifests record the engine.
- `--max-cells` for the card you get (measure peak memory on a large bundle;
  leave ≥ 15% headroom). With `_dual/_quad` partitions, `--devices
  cuda:0,cuda:1,…` uses the tiled solver (`crestimap/tiled.py`).

### 2. GPU job script (`oscer/gpu_worker.sbatch`, new)

```bash
#SBATCH -J crest-gpu
#SBATCH -p <h100 partition>[,<fallback gpu partition>]   # SLURM picks the first that can start
#SBATCH --gres=gpu:1  --cpus-per-task=8  --mem=96G
#SBATCH -t 48:00:00  --signal=B:USR1@1800
#SBATCH -o <shared>/logs/%x-%j.log
# env (HF_TOKEN from the chmod-600 file, HF_HOME, EVENT_DEM_CACHE, CREST_AI_DIR, PYTHON)
# Mode A: python -u -m crestimap.worker --sessions --publish \
#           --max-hours 47 --idle-exit-min 20 --checkpoint-dir <shared>/sessions \
#           --max-cells <measured> --work-root <shared>/jobs
```
Run from a **snapshot** of the code (the dispatcher copies the fork + CREST_AI
checkouts into `<shared>/releases/<stamp>/` before `sbatch`), so a later
`git pull` never changes files under a running job.

### 3. Dispatcher (`oscer/dispatcher.py`, new) — runs forever on the CPU node

Read-only toward Hugging Face (Mode A). Every 60 s:
1. List `events/queue/` (HfApi `list_repo_files`, `repo_type="dataset"`).
   "Claimable work" = specs with a forcing tar, a `domain`, `expected_cells ≤`
   your `--max-cells`, no `.failed.json`, and no FRESH claim (hb < 600 s) by
   another worker. Fresh claims by your own running job count as in progress.
2. Your jobs: `squeue -u $USER -n crest-gpu -h -o "%i %T"`.
3. If claimable work > 0 and no `crest-gpu` job is RUNNING or PENDING →
   refresh code (`git pull` fork v2 + CREST_AI main, snapshot) and `sbatch`.
   Optional second job only when backlog ≥ `BACKLOG_PER_JOB` × running and
   running < `MAX_JOBS` (default `MAX_JOBS=1`; raise only with the user's OK).
4. Never `scancel` a running job. Log every decision with the reason
   (`<shared>/logs/dispatcher.log`, rotated).
5. Write a status JSON every minute (`<shared>/dispatcher_status.json`:
   queue counts, jobs, last submit, last error) for the daily review.

Keep it **stateless** (everything derived from the queue + `squeue`), so a
restart is harmless. Run it under `tmux`/`screen` or `nohup` with a supervising
loop that restarts it after a crash (back off 60 s); if OSCER allows `cron`/
`scrontab` on that node, add a 10-minute "is it alive?" check that restarts it.

### 4. Mode B (only if GPU nodes have NO internet)

Then the dispatcher does all network I/O and the GPU job computes offline:
- dispatcher: claim + heartbeat (every 240 s, while the job is pending or
  running), download the bundle, pre-fetch the 3DEP DEM tiles into the shared
  `EVENT_DEM_CACHE` (call `crestimap.dem` on the CPU node), stage
  `<shared>/inbox/<event>/` (spec + extracted bundle), and later publish the
  GPU job's `outbox/<event>/` with `eventsim._update_archive`,
  `eventsim._make_pngs`, `eventstore.publish_event` after re-checking
  `_spec_unchanged`; then `_cleanup_queue` / `_report_failure` as the worker
  does today.
- GPU job: loops over `inbox/`, keeps sessions resident + checkpointed exactly
  as in §1, writes `outbox/<event>/` (frames, manifest, `done` marker), never
  touches the network.
Report which mode you implemented and why.

## Validation gates (record everything in `docs/OSCER_VALIDATION.md`)

Use **local replays** (download a queue bundle into a scratch dir and drive
`EventSession` directly) for G1–G2 — the worker writes claims even with
`--no-publish`, so shadow runs against the live queue would block the ZeroGPU
worker.

- **G0 environment:** Step-0 results; CUDA test-suite all pass; HF write probe.
- **G1 checkpoint continuity:** one bundle solved continuously vs. solved in
  3 slices with a full **process restart** and checkpoint restore between
  slices. Gate: wet-cell Jaccard ≥ 0.99 and mean |Δdepth| < 2 cm (bit-identical
  expected). Same for a catch-up visit restored from a checkpoint.
- **G2 deadline/signal:** `--max-hours 0.25` (and separately `scancel
  --signal=USR1`) on a replay: clean exit within one chunk, checkpoint present,
  claims released (in replay: the release call made), a second run resumes and
  matches the uninterrupted result (G1 thresholds).
- **G3 dispatcher:** `--dry-run` against the live queue for ≥ 1 h: decisions
  logged, no duplicate submissions, correct handling of a fresh foreign claim
  (ZeroGPU) and of an empty queue.
- **G4 live:** dispatcher on, `MAX_JOBS=1`, `--publish`. First real publish:
  manifest `provenance.engine` = your ident and `provenance.crestimap` = the
  fork SHA you ran; frames listed == PNGs stored; `maxdepth` present;
  `events/index.json` still contains every id in `events/pinned.json` that it
  contained before, with frames intact; the queue entry consumed only after a
  successful publish. Then a 48 h soak: at least one job reaching its idle exit
  or deadline and the next job resuming from checkpoints.
- **Throughput:** cell-steps/s and GPU-min per (M cells × sim-h) on the OSCER
  GPU, for the cost model and for `--max-cells`.

## Deliverables

1. Code on fork `v2`: worker flags (§1), `oscer/gpu_worker.sbatch`,
   `oscer/dispatcher.py`, `oscer/README.md` (setup, start/stop, logs, how to
   pause: stop the dispatcher; running jobs finish their current chunk and exit
   on SIGUSR1 via `scancel --signal=USR1 <jobid>`).
2. `docs/OSCER_VALIDATION.md` with G0–G4 results and the GO/NO-GO line.
3. A short message for the user: mode chosen, partitions used, measured
   throughput, the `--max-cells` you set, and anything that needs their
   decision.

Notes for the user's side (do NOT change these yourself — report them):
- Once OSCER solves whole basins, the runner's `EVENT_QUEUE_MAX_CELLS`
  (currently 16 M: big basins are trimmed for ZeroGPU) can go back to 0 so
  bundles carry the whole basin again. That is a Space variable; the user
  decides.
- ZeroGPU keeps running; exclusive claims keep the two from duplicating work.
