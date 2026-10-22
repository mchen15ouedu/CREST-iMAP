# Jetstream2 GPU worker for the 2-D event queue: runbook

**Readers:** the PI, working from an SSH shell on the new VM, and any Claude
Code session started on that VM to help. Treat this file as that session's
brief. The kit is in `deploy/jetstream2/` (file map at the end).

**The job:** a persistent Jetstream2 GPU VM (NSF ACCESS allocation EES260210)
that pulls CREST-iMAP v2 inundation jobs from the queue. It replaces the
released HPC workstation `server2017.hydros.int`. The VM runs the existing
headless worker, `python -m crestimap.worker --sessions --publish`, under
systemd. No new solver or queue code is involved.

## Ground rules (same as P1_WORKER_BRIEF.md / OSCER_DISPATCH_BRIEF.md)

- Never push to CREST_AI. Never change a Hugging Face Space, Space variable
  or secret.
- Write to the dataset `vincewin/CREST_data` only through the worker:
  the queue protocol plus `eventstore.publish_event`. Never edit queue
  files or event folders by hand. Never touch `events/pinned.json`, which
  holds the user's demo set.
- **Basins, never rectangles. Resolution is never coarsened.** The worker
  leaves a job it cannot hold unclaimed; it never shrinks or coarsens it.
  No runoff bias scaling.
- **The HF token** lives only in `/etc/crest-worker/hf_token` (root:root
  0600). It must never be put in git, in the env file, on a command line, in
  shell history or in a log. Do not run `huggingface-cli login` on the VM.
- **No shadow runs against the live queue.** `--no-publish` still writes and
  heartbeats claims, which blocks the ZeroGPU worker. Validate offline with
  `replay_bundle.py` (gate G1). The service accepts only `--poll-s N` in
  `WORKER_EXTRA_ARGS` and refuses everything else, including `--no-publish`
  and `--once`.
- Changing solver physics (`FORK_REF`) is the user's decision. So is the
  runner's `EVENT_QUEUE_MAX_CELLS`.

## Decisions baked into the kit

| Item | Choice | Why |
|---|---|---|
| Flavor | **g3.large**: 16 vCPU, 60 GB RAM, half A100 (20 GB vGPU), 32 SU/h | The queue caps jobs at 16 M bbox cells, which need about 4.7–6.6 GB. The largest job ever solved (27.8 M, `2026101123_09426620`) needs about 10.7 GB. The card holds about 49 M cells at 350 B/cell with 15 % headroom. 200,000 SU lasts about 6,250 h, roughly 260 days of 24/7 running. |
| Image | `Featured-Ubuntu24` | Ships NVIDIA drivers and is rebuilt weekly. `Featured-Minimal-Ubuntu24` also works: it has the same NVIDIA drivers but no desktop, so it uses less of the root disk. The kit also supports Ubuntu 22.04. |
| Disk | 60 GB root disk (flavor default) for the OS, venv and releases; a **300 GB volume** at `/media/volume/crest-data` for the data | Every allocation includes 1 TB of Jetstream2-Storage by default; nothing has to be requested. The DEM tiles, HF cache and job directories go on the volume instead of sharing 60 GB with the OS image, the torch venv and the releases. A full disk mid-visit raises an exception, and the worker then consumes the job as `failed.json`. A daily timer prunes the caches. |
| Solver code | fork commit `c5184759`, pinned (`FORK_REF`) | The physics the PI approved on 2026-10-20 (draining-time outflow limiter + interflow store), also run by the ZeroGPU worker. `update.sh` refuses a `FORK_REF` whose solver-path files differ from it unless `ALLOW_SOLVER_CHANGE=1`, and anything older than `4c55c0f6` (the CPU Spaces' `CRESTIMAP_REF` and every HPC manifest) is refused outright. |
| hf_data | GitHub `CREST_AI` `main` (≥ `731ab17`), refreshed daily | Main is meant to be identical to the deployed Spaces, and the OSCER brief asks for a fresh `main` before GPU work. `crest-worker-update.timer` rebuilds from `main` every day and restarts the worker between visits when `hf_data/` changed. `healthcheck.sh` compares the publish-path modules (`eventstore`, `eventsim`, `eventqueue`, `provenance`) with `main` and with the `vincewin/CREST_demo` Space, which the ZeroGPU worker uses. Every release is refused if `eventstore.py` lacks `def pinned_ids`. The obsolete HPC `space_ref` copy and `sync_space_ref.py` are not used. |
| Cell budget | `EVENT_MAX_CELLS_GPU=40000000` | Above the 16 M queue cap, below the card's ceiling. Re-set it from the G1 measurement. |
| Engine label | `jetstream2:<hostname>` | Stamped as `provenance.engine` in manifests and shown raw on the dashboard. Claims read `<hostname>:<pid>`, because `crestimap.worker` hard-codes that format. |

When to use **g3.xl** instead (32 vCPU, full A100 40 GB, 64 SU/h, about 130
days on the budget): only if the user raises `EVENT_QUEUE_MAX_CELLS` above
about 45 M. On g3.xl set `EVENT_MAX_CELLS_GPU=90000000` (ceiling about 101 M).
Untrimmed basins reach about 120 M bbox / 44.6 M active cells, which needs
35–43 GB plus context. That does not fit even g3.xl with headroom; it would
need a worker code change for single-device tiling. Switching flavors is
covered under Day-2.

## 1. Prerequisites (PI)

1. The ACCESS exchange of EES260210 credits for **Jetstream2 GPU** SUs is
   approved. Approval can take up to a week.
2. Check, rather than change: in the ACCESS Allocations portal
   (https://allocations.access-ci.org), open EES260210 and its user
   management page (the label may differ) and confirm that your own row has
   the Jetstream2 GPU resource. The PI is normally added to Jetstream2
   resources automatically; other users must be added by the PI, a co-PI or
   an allocation manager. After any change, **wait up to 4 h** before
   Exosphere shows the allocation.
3. An SSH key on your laptop, for example
   `ssh-keygen -t ed25519 -f ~/.ssh/js2_crest`.
4. Access to the Hugging Face account that owns `vincewin/CREST_data`, so you
   can create the token in step 5.
5. Check that g3.large is available on the "Availability of Scarce
   Resources" dashboard (https://docs.jetstream-cloud.org/overview/status/).
   The large GPU flavors regularly run out.

## 2. Create the VM

**Exosphere** (https://jetstream2.exosphere.app/):

0. First time only: on the home page click **Add Allocation**, then **Add
   ACCESS account**. At CILogon choose **ACCESS CI (XSEDE)** as the identity
   provider and complete Duo. Exosphere then lists your allocations: select
   EES260210 (unselected allocations do not appear).
1. Click the **EES260210** allocation card, then **Create**, then **Instance**.
2. Choose **By Image** and pick `Featured-Ubuntu24` (or
   `Featured-Minimal-Ubuntu24`).
3. Name: `crest-js2-gpu1`. The name becomes the hostname, the claim ident
   (`crest-js2-gpu1:<pid>`) and the engine label
   (`jetstream2:crest-js2-gpu1`).
4. Flavor: **g3.large**.
5. Root disk size: the default (60 GB).
6. Quantity 1. Web desktop off.
7. Under Advanced Options:
   - Install OS updates: yes.
   - Guacamole: optional; the Web Shell is a handy fallback.
   - Network: default.
   - Public IP: Automatic.
   - **SSH Public Key**: upload `~/.ssh/js2_crest.pub`.
   - Leave the Boot Script alone.
8. Click **Create**. Wait for status Ready (green) in several minutes. A
   g3.xl takes longer because it installs its driver at first boot.
9. **Data volume:** **Create**, then **Volume**: name `crest-data`, size
   300 GB. Attach it to `crest-js2-gpu1`. Exosphere then shows its mount
   point, normally `/media/volume/crest-data`; confirm it on the VM with
   `findmnt /media/volume/crest-data`. If the path differs, use that path
   wherever this runbook says `/media/volume/crest-data`. The volume counts
   against the allocation's default 1 TB of Jetstream2-Storage.

**CLI alternative.** Get an application-credential `openrc` from Horizon
(https://js2.jetstream-cloud.org): Identity, then Application Credentials,
then Create.
- **Set an expiration date**, for example the allocation's end date. If you
  leave it blank, the expiry defaults to today, and every
  later CLI step (shelve, unshelve, floating IP, recreating the instance)
  fails with an authentication error.
- Keep the secret: it cannot be retrieved later.
- Use only the openrc downloaded for this application credential. The
  openrc generator in Horizon's username menu (top right) does not work with
  Jetstream2.

Then:

```bash
source ~/EES260210-openrc.sh
openstack keypair create --public-key ~/.ssh/js2_crest.pub crest-key
openstack security group create crest-ssh --description "SSH + ICMP"
openstack security group rule create --protocol tcp --dst-port 22:22 --remote-ip <your-ip>/32 crest-ssh
openstack security group rule create --protocol icmp crest-ssh
openstack server create crest-js2-gpu1 --flavor g3.large --image Featured-Ubuntu24 \
  --key-name crest-key --security-group crest-ssh --nic net-id=auto_allocated_network --wait
openstack floating ip create public          # note the address
openstack server add floating ip crest-js2-gpu1 <address>
openstack volume create --size 300 crest-data
openstack server add volume crest-js2-gpu1 crest-data
```

Check on the VM whether the CLI-attached volume is mounted (`lsblk`,
`findmnt /media/volume/crest-data`). If it is not, format it once
(`sudo mkfs.ext4 /dev/<device>`, only if `lsblk -f` shows no filesystem),
mount it at `/media/volume/crest-data`, and add an `/etc/fstab` line with
the `nofail` option.

If Exosphere has never been used on this allocation,
`auto_allocated_network` may not exist yet. Create a network, subnet and
router, and set the router's external gateway to `public`, as described at
https://docs.jetstream-cloud.org/ui/cli/network/.

## 3. SSH in

```bash
ssh -i ~/.ssh/js2_crest exouser@<public-ip>     # Exosphere-created VM
ssh -i ~/.ssh/js2_crest ubuntu@<public-ip>      # CLI-created VM
```

Both users have sudo. Jetstream2 also creates a DNS name of the form
`crest-js2-gpu1.<allocation>.projects.jetstream-cloud.org`, pointing at the
public IP. This works only for an instance on `auto_allocated_network` with
a public IP, and the docs give no time for it to appear; use the IP until it
resolves.

Optional firewall (Exosphere's default security group allows all inbound
traffic):

```bash
sudo ufw allow ssh
sudo ufw allow 49528/tcp     # only if you use the Web Shell
sudo ufw enable
```

## 4. Get the kit onto the VM

**Preferred: clone it from GitHub.** This works once the kit commit is on
fork `v2`. The kit touches no `crestimap/` file, so pushing it changes no
Space.

```bash
git clone --filter=blob:none --sparse -b v2 https://github.com/mchen15ouedu/CREST-iMAP.git ~/crest-kit
git -C ~/crest-kit sparse-checkout set deploy/jetstream2 docs
```

**If the kit is not pushed yet**, copy it from the laptop:

```bash
ssh exouser@<ip> mkdir -p crest-kit/deploy crest-kit/docs
scp -r deploy/jetstream2 exouser@<ip>:crest-kit/deploy/
scp docs/JETSTREAM2_WORKER.md exouser@<ip>:crest-kit/docs/
ssh exouser@<ip> "sed -i 's/\r$//' crest-kit/deploy/jetstream2/*"   # strip Windows CRLF, if any
```

## 5. Bootstrap

```bash
# with the data volume from section 2 (recommended):
sudo INIT_DATA_ROOT=/media/volume/crest-data bash ~/crest-kit/deploy/jetstream2/bootstrap.sh
# without a volume (everything on the 60 GB root disk):
sudo bash ~/crest-kit/deploy/jetstream2/bootstrap.sh
```

`INIT_DATA_ROOT` matters only on the first run, when bootstrap creates the
env file. To move the data later, see Day-2 "Disk".

It is idempotent: re-run it any time. It never enables or starts the worker.
It does the following, in order:

1. Checks `nvidia-smi`, the driver (≥ 525) and the GPU memory (≥ 9.5 GB).
   Prints the virtualization mode and, on a vGPU flavor, the vGPU license
   status (warns unless it is `Licensed`).
2. Installs `git curl python3-venv`.
3. Creates the system user `crest` (home `/var/lib/crest-worker`).
4. Creates `/etc/crest-worker/crest-worker.env` from the example. It never
   overwrites an existing file.
5. Creates the data directories. It refuses a path under `/media/volume/`
   while no volume is mounted there.
6. Creates a venv at `/opt/crest-worker/venv` and installs `torch>=2.4,<2.15`
   from the **CUDA 12.6** wheel index. A plain `pip install torch` would pull
   CUDA 13 wheels, which need driver 580+. It also installs numpy, rasterio,
   requests, pyarrow, pillow, huggingface_hub and pytest. pip keeps no
   download cache.
7. Runs a GPU matmul as `crest`.
8. Builds the first release under `/opt/crest-worker/releases/<UTC stamp>/`:
   - the fork at `FORK_REF`, `crestimap/` only;
   - CREST_AI `main`, `hf_data/` only.
9. Verifies the release (production-pin ancestry, solver-path guard, CREST_AI
   ≥ `731ab17`, `pinned_ids` and the KEEP_FULL clamp) and import-tests it as
   `crest`. It refuses to continue on any failure. It then reports whether
   the release's publish-path `hf_data` matches GitHub `main` and the
   `vincewin/CREST_demo` Space (report only).
10. Installs the scripts into `/opt/crest-worker/bin` and the units into
    `/etc/systemd/system`. It enables only the two timers: the daily cache
    prune, and the daily code refresh (unless `AUTO_UPDATE=0`).
11. Probes outbound access to huggingface.co, the 3DEP S3 bucket and GitHub.

Review `/etc/crest-worker/crest-worker.env` afterwards. It holds no secrets,
only the pins, the cell budget and the paths. If you edit it later, run
`sudo /opt/crest-worker/bin/update.sh --force` (code changes) or
`sudo systemctl restart crest-worker` (env only).

## 6. The token (PI only)

1. On huggingface.co, signed in as the account that owns
   `vincewin/CREST_data`, go to **Settings**, then **Access Tokens**, then
   **Create new token**.
   - Token type: **Fine-grained**. Name: `crest-js2-worker`.
   - Under **Repositories permissions**, add `vincewin/CREST_data` (the
     dataset). Tick **Read access to contents of selected repos** and **Write
     access to contents/settings of selected repos**.
   - Tick nothing else: no user, org, inference or other-repo permissions.
     The worker reads and writes this one public dataset. GitHub and the
     3DEP bucket are anonymous.
   - Create the token and copy it once.
2. On the VM:

   ```bash
   sudo /opt/crest-worker/bin/bootstrap.sh --set-token
   ```

   It asks for the token with hidden input, so the token never appears on
   screen, on a command line or in shell history. It writes the token
   atomically to `/etc/crest-worker/hf_token` (root:root 0600), then runs the
   read-only `whoami` scope check. Expect `role=fineGrained`, a single scoped
   dataset `vincewin/CREST_data` with write, and `token scope: OK`. Re-check
   any time with `sudo /opt/crest-worker/bin/bootstrap.sh --check-token`.

How the service gets the token: systemd's `LoadCredential=` copies the
root-only file into `/run/credentials/crest-worker.service/`, which only the
service can read. `crestimap.worker` reads `~/huggingface.txt` when `HF_TOKEN`
is unset, and bootstrap makes that file a symlink to the credential. The
token is therefore never in the unit, the env file or `systemctl show`. The
unit also refuses to start while the token file is empty
(`AssertFileNotEmpty=`).

**Ubuntu 22.04 (systemd 249):** `ExecStartPre=` sees `$CREDENTIALS_DIRECTORY`
set but empty; only `ExecStart=` and `ExecStopPost=` see the credential. This
was verified on crest-js2-gpu1 on 2026-10-22.
- `run-worker.sh` therefore repoints and checks the symlink itself.
- `preflight.sh` only notes the deferral.
- The start-time `release_claims.py` run (in `ExecStartPre=`, best effort) has
  no token there, so it cannot release the claims of a crashed run. The
  `ExecStopPost=` run, which sees the credential, normally releases them. Any
  claim left behind goes stale after 600 s.
- systemd 255 on Ubuntu 24.04 has none of this.

## 7. Validation gates

Record every gate in `docs/JETSTREAM2_VALIDATION.md` in the VM's kit checkout,
using the template at the end of this section. Run G0 and G1 before the
worker ever runs. **G2 is the first time the VM writes to the live dataset,
so get the user's go-ahead first.**

### G0: environment and GPU

```bash
nvidia-smi
nvidia-smi -q | grep -iE 'Virtualization Mode|MIG'
nvidia-smi -q | grep -iA2 'vGPU Software Licensed Product'   # g3.large and smaller
sudo /opt/crest-worker/bin/bootstrap.sh --check-token
sudo /opt/crest-worker/bin/crest-py CRESTIMAP_TEST_DEVICE=cuda -m pytest -q -p no:cacheprovider \
  --ignore=crestimap/tests/test_burn_screen.py crestimap/tests
sudo /opt/crest-worker/bin/healthcheck.sh
```

`test_burn_screen.py` loads `tools/burn_channels.py`, a DEM-burn tool that is
not in a release (releases carry `crestimap/` only) and that the worker never
calls. Without `--ignore`, collection stops with FileNotFoundError.

Known CUDA test-harness failures at `c5184759`, first seen on crest-js2-gpu1 on
2026-10-22:
- `test_forcing.py` and `test_solver.py` call `torch.set_default_device(DEVICE)`
  at import. In one pytest process that default leaks into the files after
  them, so 7 tests in `test_event.py`, `test_session.py` and `test_tiled.py` fail
  with "can't convert cuda:0 device type tensor to numpy". Run those three
  files in their own processes, for example
  `... -m pytest -q -p no:cacheprovider crestimap/tests/test_session.py`. All 9
  passed there.
- `test_solver.py::test_well_balanced_steep_terrain` draws `torch.rand` from a
  CPU generator while the default device is cuda, so it fails on CUDA in
  every setup.

Both are test-setup bugs. The worker never sets a default device, and the G1
replay drives the same session code on the GPU.

Pass criteria:
- The driver is ≥ 525, and its version is recorded.
- On a vGPU flavor (`Virtualization Mode : VGPU`, i.e. g3.large):
  `License Status : Licensed (...)`. An unlicensed vGPU runs CUDA degraded
  after 20 minutes and fails every CUDA call after 24 hours; the preflight
  refuses to start the worker on one. Not applicable on g3.xl
  (`Pass-Through`, generic driver).
- torch sees the GPU (bootstrap printed `matmul ok`).
- **The CUDA test suite passes completely.** `test_session.py` is the slow
  one; the HPC took 30 min while shared.
- The token scope is OK.
- Outbound HTTP to HF, S3 and GitHub works (bootstrap's probe lines).
- The healthcheck shows nothing wrong except "not enabled: expected before
  gate G2". Its "hf_data drift" section reads `publish-path hf_data in
  sync`; if the Space differs from GitHub `main` in a publish-path module,
  tell the user before G2.

### G1: offline solve of an already-published bundle

No token, no claim, no publish. `replay_bundle.py` drops `HF_TOKEN` and
disables implicit tokens, so it cannot write. It drives `EventSession`
exactly as `Worker.process_visit` does for a new session, then
`eventsim._update_archive` and `_make_pngs`.

```bash
# the largest event ever published (HPC, 27.8 M bbox cells, crestimap 4c55c0f6)
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/replay_bundle.py \
  --event 2026101123_09426620 --find-revision
```

`--find-revision` uses the commit that last wrote `events/<id>/manifest.json`.
The worker publishes before it consumes the queue entry, so the bundle behind
the published result is still queued at that commit (for this event:
`65a905c1fd38`). If that fails, the tool falls back to a slow scan of the
commit history. `--revision <sha>` skips the lookup. A bundle that is queued
right now replays from HEAD with no revision; list the queued bundles with
`queue_status.py`.

The report is written to
`$DATA_ROOT/replay/<id>/replay_report.json` (`/media/volume/crest-data/...`
with the volume, `/var/lib/crest-worker/data/...` without).

Pass criteria:
- The tool exits 0 and reports `native_resolution: true`. The grid equals the
  spec's `n_bbox` and `dx_m` ≤ 45; the tool fails hard on any coarsening.
- `published_comparison.same_grid` is true.
- `peak_fraction_of_gpu` ≤ 0.85.
- `frames` > 0, or `dry` matches the published event.
- `frame_wet_jaccard` and `frame_mean_abs_ddepth_cm_wet` are recorded. They
  compare the last depth frame both runs wrote. Expect at least about 0.9.
  They are informational, because the published run continued a resident
  session with momentum; investigate anything below 0.8.
  `episode_maxdepth_*` partly shares the published archive by construction.
- No commit from this host appears in the dataset history.

Act on the report:
1. Set `BYTES_PER_CELL_EST` to `suggested_BYTES_PER_CELL_EST`.
2. Set `EVENT_MAX_CELLS_GPU` to at most `suggested_EVENT_MAX_CELLS_GPU`, and
   never below 16 M (the queue cap).
3. Record `gpu_min_per_Mcell_simh`. The RTX 4500 Ada did 1.0–1.45.

### G2: first live claim and publish (needs the user's go-ahead)

```bash
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py \
  --pins save /var/lib/crest-worker/pins_baseline.json      # pinned demo set, before
sudo systemctl enable --now crest-worker
journalctl -u crest-worker -f
```

Expect these, in order:
1. `preflight: OK`.
2. `run-worker: release ... engine=jetstream2:<host>`.
3. `worker <host>:<pid> up (P1.6 sessions) ... publish=ON`.
4. `claiming <id>`.
5. Dataset commits `claim <id> [<host>:<pid>]` about every 240 s during the
   solve.
6. `<id>: publish OK`.
7. A commit `worker done <id> (consumed)`.

The HPC's stale claim on `2026101511_07239500` (15.0 M bbox) is taken over
through the normal stale-claim path. It is probably the first job.

Then verify:

```bash
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py --verify-publish <id>
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py \
  --pins check /var/lib/crest-worker/pins_baseline.json
sudo /opt/crest-worker/bin/healthcheck.sh
```

Pass criteria:
- In the manifest:
  - `provenance.engine` = `jetstream2:<host>`;
  - `provenance.crestimap` = the release's fork SHA (`c518475956c9`);
  - `provenance.crest_ai` = the release's CREST_AI SHA, not null as on the HPC.
- Every listed frame's tif and png is stored, and `maxdepth` is present.
- The grid is native and has a basin domain.
- The `index.json` entry matches.
- The pins check **PASS**es: every pinned id that was listed is still listed,
  with its frames.
- The queue entry was consumed only after `publish OK`.
- The event appears on the dashboard.

If the claim commit fails with 401/403, the token lacks write scope. Fix the
token; nothing was changed.

### G3: heartbeat and claim release on `systemctl stop`

Run this while the worker holds a claim. Use a small job if you can choose,
because the interrupted visit re-runs.

```bash
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py --own   # note hb
sleep 300; sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py --own   # hb advanced
time sudo systemctl stop crest-worker
journalctl -u crest-worker -n 40 --no-pager | grep -E 'KeyboardInterrupt|release_claims'
sudo /opt/crest-worker/bin/crest-py /opt/crest-worker/bin/queue_status.py   # entry still queued, unclaimed
sudo systemctl start crest-worker                                           # re-claims, finishes
```

Pass criteria:
- The heartbeat age stays below 240 s and changes between the two reads.
- The stop finishes within `TimeoutStopSec` (180 s).
- `release_claims: <id>: released` appears.
- The spec and bundle are still queued: not consumed, and no `failed.json`.
- After the start, the job is re-claimed and published.

Optional extra checks:
- `sudo systemctl kill -s SIGKILL crest-worker`: the claim is still released
  by `ExecStopPost`, and the service restarts after 120 s.
- `sudo reboot` while idle: the service comes back by itself, because the
  preflight waits up to 5 min for the GPU driver.

Background: the worker has no signal handler. `KillSignal=SIGINT` makes it
unwind through KeyboardInterrupt, its `finally` stops the heartbeat thread,
and `release_claims.py` deletes the leftover `.claim` through the worker's
own `Worker._cleanup_queue(success=False)`.

### G4: 24 h soak

```bash
sudo /opt/crest-worker/bin/healthcheck.sh --since 24h
```

Pass criteria: the summary is `HEALTHY`, which means:
- 0 `visit FAILED`, 0 CUDA out-of-memory errors, 0 `publish FAILED`;
- no restarts apart from G3's and a daily-update restart recorded in
  `updates.log`;
- no hf_data drift, and the last automatic update did not fail;
- no claims left by dead processes;
- the pins check PASSes;
- disk free is above `DISK_MIN_FREE_GB`;
- `publish OK` was logged for every job taken.

Also run `--verify-publish` on one more event. Expect about 768 SU used
(32 SU/h × 24). Write the GO/NO-GO line.

### Template: `docs/JETSTREAM2_VALIDATION.md`

```markdown
# Jetstream2 GPU worker - validation
VM: <name> (<flavor>), image <image>, driver <ver>, GPU <name / vGPU profile>, <MiB>
Virtualization mode <VGPU|Pass-Through>; vGPU License Status <Licensed (Expiry ..)|n/a>
Data: <volume name, size, mount point | root disk>
Release: fork <sha12> (FORK_REF <ref>), CREST_AI <sha12>; torch <ver> (CUDA <ver>)
hf_data drift at G0: <none | details>; update timer: <enabled|disabled>, next <..>
Dates: G0 <..> G1 <..> G2 <..> G3 <..> G4 <..>
## Verdict: GO | NO-GO  (<one line>)
| gate | criterion | result |
|---|---|---|
| G0 | vGPU licensed (or passthrough); CUDA test suite all pass; token scope OK; outbound OK; no hf_data drift | |
| G1 | native grid; same grid as published; peak <= 85 %; J (info) | id, M cells, peak GiB, B/cell, GPU-min/(Mcell*h), J |
| G2 | first publish verified; pins PASS; consumed after publish | id, commit links |
| G3 | hb < 240 s; stop < 180 s; claim released, entry kept; re-claimed | |
| G4 | 24 h HEALTHY | counts |
Settings after G1: EVENT_MAX_CELLS_GPU=<..> BYTES_PER_CELL_EST=<..>
Notes / anything the user must decide:
```

## 8. HPC handoff (server2017)

The HPC's `hf_data` is pre-`939800a`. It has no `pinned.json` support and
**would strip the pinned demo animations from about 2026-10-29**. Its last
heartbeat was 2026-10-19 19:09Z, but a silent heartbeat does not prove the
process is dead.

1. **After G2 passes, and no later than 2026-10-28**, SSH to server2017 and
   stop the old worker:

   ```bash
   pgrep -af crestimap.worker            # running? note the PID
   kill -TERM <pid>                      # exits at once; its claim goes stale after 600 s
   sleep 30; pgrep -af crestimap.worker || echo stopped
   kill -KILL <pid>                      # only if the PID is still listed
   pgrep -af crestimap.worker || echo stopped   # must say "stopped"
   crontab -l | grep -i worker           # nothing may restart it (also check tmux/screen loops)
   ```

   Do **not** use `kill -INT` here: it does nothing to this process.
   `run_worker.sh` started it as `nohup "$PYTHON" -u -m crestimap.worker ... &`
   from a non-interactive script. Bash starts such background commands with
   SIGINT ignored, and Python only turns SIGINT into KeyboardInterrupt when it
   inherits the default handler. The worker installs no SIGTERM handler, so
   SIGTERM ends it immediately. Its claim is not released; it goes stale after
   600 s, which is the intended outcome (see step 3). The Jetstream2 service
   is different: systemd starts it with default signal handlers and
   `run-worker.sh` uses `exec`, so its `KillSignal=SIGINT` works.

   If G2 has not passed by 2026-10-28, stop the HPC worker anyway. A gap in
   GPU coverage is better than losing the demo set; ZeroGPU covers small jobs.
2. Remove the HPC's token file (`~/huggingface.txt`) and revoke that token in
   the HF settings, so that no stray process can publish with old code.
3. Do not delete its leftover claim by hand. Stale claims are re-claimable by
   design, and the Jetstream2 worker takes them over.

## 9. Day-2 operations

**Health:** run `sudo /opt/crest-worker/bin/healthcheck.sh`. It is
read-only, exits 1 on problems and reports:
- the GPU, and on a vGPU flavor its license status;
- the service state and restarts, and the update timer;
- code versions, including whether the running process uses the current
  release;
- hf_data drift: the release's publish-path modules against GitHub `main`
  and against the `vincewin/CREST_demo` Space;
- the queue and claims, the latest publish per engine, and the pinned demo
  set against the G2 baseline;
- disk, the SU estimate and recent logs.

`--quick` skips Hugging Face and GitHub.

**Logs:**
- `journalctl -u crest-worker -f` to follow the worker.
- `journalctl -u crest-worker --since -6h | grep -E 'claiming|publish|FAILED|skip'`
  for a quick history.
- `journalctl -u crest-worker-prune` for the cache housekeeping.
- `journalctl -u crest-worker-update` and `/var/lib/crest-worker/updates.log`
  for the code refreshes.

**Updating code** (newest CREST_AI `main` hf_data; the fork stays at
`FORK_REF`). This happens **automatically every day**:
`crest-worker-update.timer` (about 14:00–16:00 UTC) runs `update.sh --auto`.
It builds and verifies a new release and compares it with the running one.
- If the fork SHA and the `hf_data/` tree are unchanged, nothing restarts.
  A CREST_AI commit outside `hf_data/` therefore costs no restart, which
  matters because resident sessions live only in RAM.
- Otherwise it waits up to 45 min for the worker to hold no claim, then
  stops it, switches `current` and starts it. If the worker is still busy,
  the run is deferred to the next day without error.
- If the new release does not stay up, it switches back to the previous
  release, starts that, and the update unit ends as failed (healthcheck
  reports it).

To opt out, set `AUTO_UPDATE=0` in the env file and re-run bootstrap. By hand:

```bash
sudo /opt/crest-worker/bin/update.sh            # build, verify, wait until no claim is held, stop, switch, start
sudo /opt/crest-worker/bin/update.sh --check    # build + verify only (and the drift report)
sudo /opt/crest-worker/bin/update.sh --rollback # back to the previous release
```

A running Python process keeps the modules it imported. This is why the HPC
ran weeks-old code. `git pull` alone therefore does nothing: the switch to a
new release is always a restart. Each release is an immutable directory
(`releases/<stamp>`). `run-worker.sh` resolves `current` once at start, and
the last three releases are kept.

**hf_data drift.** The ZeroGPU worker takes `hf_data` from the CREST_demo
Space, not from GitHub, and the CPU Spaces run their deployed copy. If
healthcheck reports that a publish-path module (`eventstore`, `eventsim`,
`eventqueue`, `provenance`) differs between this release and the Space,
this VM would publish and apply retention with different code than the rest
of the system. Run `update.sh`. If the drift is between GitHub `main` and
the Space themselves, tell the user, who decides which one is right.
Differences in other modules are listed for information only.

**Kit changes:** after the scripts or unit change, update the kit checkout
(`git -C ~/crest-kit pull`), re-run `sudo bash
~/crest-kit/deploy/jetstream2/bootstrap.sh`, then run `sudo
/opt/crest-worker/bin/update.sh --force`.

**Changing the solver commit:** this is a physics change and the user's
decision. The kit defaults to `c5184759`, approved on 2026-10-20. To move to
a later commit after the user approves it:
1. Set `FORK_REF=<sha>` and `ALLOW_SOLVER_CHANGE=1` in the env file.
2. Run `update.sh`.
3. Repeat G1 on the new code and note it in the validation file.
4. Ask for `SOLVER_APPROVED_SHA` in `deploy/jetstream2/lib.sh` to be bumped in
   the repo, then set `ALLOW_SOLVER_CHANGE=0` again.

ZeroGPU checks out fork v2 HEAD unpinned at each restart. The CPU Spaces stay
on `4c55c0f6` until their Dockerfile pin changes.

**Cell budget:**
1. Edit `EVENT_MAX_CELLS_GPU` (and `BYTES_PER_CELL_EST`).
2. Run `sudo systemctl restart crest-worker`.

The preflight refuses a budget the card cannot hold. That matters because a
CUDA out-of-memory error consumes the job as `failed.json`.

**SU burn:**
- Rates: g3.large is 32 SU/h (768 per day); g3.xl is 64 SU/h.
- States: active is charged in full, even when idle. Stopped is charged
  50 %. Shelved is free. **Never suspend a GPU instance** (known libvirt
  issue).
- Where to look: the real balance is on the Exosphere allocation card or the
  ACCESS portal, 12–24 h behind. `healthcheck.sh` prints an estimate; set
  `SU_BUDGET_START` in the env file for a projected runway.

**Shelve and unshelve** (to pause, at 0 SU):
1. Run `sudo systemctl stop crest-worker`. This releases claims while the
   network is still up.
2. With the data on a volume: detach `crest-data` in Exosphere (Jetstream2
   recommends detaching volumes before shelving).
3. In Exosphere, open the instance and choose Actions, then **Shelve**.
4. To resume, choose Actions, then **Unshelve**, and wait for Ready.
5. With a volume: attach `crest-data` again and confirm the mount with
   `findmnt /media/volume/crest-data`. The service is enabled, so it tried to
   start at boot. With the volume missing, the preflight refused
   ("no volume is mounted there") and the service retried until
   `start-limit-hit`. Once the volume is mounted, run
   `sudo systemctl reset-failed crest-worker && sudo systemctl start crest-worker`.
   Without a volume the service starts by itself.
6. Run `healthcheck.sh`.

Caveats:
- Unshelving a g3 instance probably needs free g3 capacity at that moment.
  The Jetstream2 docs do not say so; check the "Availability of Scarce
  Resources" dashboard before you shelve and before you unshelve.
- Public IPs of instances shelved 45 days or more may be reclaimed, and
  instances shelved for over a year may be deleted.

**Maintenance:**
- Jetstream2 GPU maintenance is the first Tuesday of each month, 07:00–19:00
  ET. The next windows are 3 Nov and 1 Dec 2026, announced at
  https://jetstream.status.io. GPU instances may be stopped and started
  during the window. The service comes back on its own; run `healthcheck.sh`
  afterwards.
- Featured images install security updates but never reboot. Reboot after
  kernel updates, at a moment when `queue_status.py --own` shows no claim.
- Jetstream2 policy allows staff to limit GPU runtime to two weeks at a time
  when resources are contended. Confirm with help@jetstream-cloud.org that a
  24/7 queue worker is acceptable.

**Disk:**
- Every allocation includes 1 TB of Jetstream2-Storage by default; nothing
  has to be requested unless more is needed. Keep the data on a volume (the
  default plan, section 2). The 60 GB root disk holds the OS image, the
  venv (torch alone is several GB) and the releases.
- The daily `crest-worker-prune.timer` removes HF download-cache blobs older
  than 2 days. It removes DEM tiles only when free space falls below
  `DISK_MIN_FREE_GB`. bootstrap installs with `pip --no-cache-dir`, so no
  pip cache builds up.
- To move the data onto a volume later:
  1. Run `sudo systemctl stop crest-worker`.
  2. In Exosphere: Create, then Volume (for example 300 GB, `crest-data`).
     Attach it and confirm the mount point with `findmnt`.
  3. In `/etc/crest-worker/crest-worker.env`, point `DATA_ROOT`,
     `WORKER_WORK_ROOT`, `EVENT_DEM_CACHE` and `HF_HOME` there (for example
     `/media/volume/crest-data/jobs`).
  4. Re-run bootstrap, then `sudo systemctl start crest-worker`. The caches
     refill by themselves; nothing has to be copied.

  Paths under `/media/volume/` are refused by bootstrap and the preflight
  while no volume is mounted there, so the data never silently lands on the
  root disk. Keep data out of `/home`, because the service runs with
  `ProtectHome=yes`.

**Switching flavor (g3.large and g3.xl):** **recreate, don't resize.**
Resizing full-card instances is discouraged, and the vGPU and passthrough
drivers differ.
1. Create the new instance with its own data volume (section 2); the caches
   refill by themselves. Delete the old volume together with the old
   instance in step 6.
2. Run bootstrap, the token step (section 6), G0 and G1 on it.
3. Set its `EVENT_MAX_CELLS_GPU` (g3.xl: 90 M).
4. Stop the old service. Its claims are released.
5. Enable the new service and repeat G2's checks.
6. Delete the old instance.

Run only one live worker per hostname.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `crest-worker.service: Start request repeated too quickly` / `start-limit-hit` | More than 6 starts in an hour. Read `journalctl -u crest-worker -n 100`, fix the cause, then `sudo systemctl reset-failed crest-worker && sudo systemctl start crest-worker`. |
| `preflight: REFUSING TO START: ...` | The message names the problem: token, retention env var, `WORKER_EXTRA_ARGS` other than `--poll-s N`, unlicensed vGPU, budget too large, release without `pinned_ids`, a `/media/volume/` path with no volume mounted, or a missing data dir. |
| `the vGPU is not licensed (...)` | The vGPU guest has no license (`nvidia-smi -q \| grep -A2 'vGPU Software Licensed Product'`). Licensing normally completes within minutes of boot; if it stays unlicensed, ask help@jetstream-cloud.org. Do not run the worker: CUDA calls fail after 24 h and each job taken would become `failed.json`. |
| `nvidia-smi` fails after a kernel update | Reboot first. If it still fails, rebuild the DKMS modules (Ubuntu 22.04: `ls /usr/lib/modules \| sudo xargs -n1 /usr/lib/dkms/dkms_autoinstaller start`) or ask help@jetstream-cloud.org. |
| torch reports no CUDA (bootstrap passed the driver ≥ 525 check) | All CUDA 12.x wheels, cu124 included, need the same driver floor (525.60.13), so switching the wheel index does not help. Check, in order: (a) the vGPU license, as above; (b) that `sudo -u crest nvidia-smi` sees the GPU and that the unit sets no `PrivateDevices=`; (c) `torch.version.cuda` against the "CUDA Version" that `nvidia-smi` prints: Jetstream2 recommends the same major revision; (d) whether the error comes from a PTX/JIT path (kernel compile at run time), which needs a newer driver than precompiled kernels. Then ask help@jetstream-cloud.org. Never use the default PyPI torch (CUDA 13) on a pre-580 driver. |
| `run-worker: refusing WORKER_EXTRA_ARGS` | Only `--poll-s N` (30–3600) is allowed. Budgets, paths and devices come from the env file, where the preflight checks them. |
| `update.sh: ... did not stay up; rolled back` (healthcheck: "a release failed to stay up") | The new release failed to start, so the previous one runs again, and the daily `--auto` run skips that exact code (`/var/lib/crest-worker/update.blocked`). Read `journalctl -u crest-worker -n 100` and tell the user (the cause is usually new hf_data on CREST_AI `main`). Once `main` is fixed, run `sudo /opt/crest-worker/bin/update.sh` by hand; a successful update clears the block. Then `sudo systemctl reset-failed crest-worker-update`. |
| `commit failed (claim ...)`: 401/403 | Token scope or revocation. Run `bootstrap.sh --check-token`, and use `--set-token` with a new token. |
| `skip <id>: ... exceeds the ... budget` | Expected for jobs above `EVENT_MAX_CELLS_GPU`. They are never coarsened and are left for a bigger worker. |
| `visit FAILED ... CUDA out of memory` | The job was consumed as `failed.json` (worker behavior). Lower `EVENT_MAX_CELLS_GPU` / raise `BYTES_PER_CELL_EST` and tell the user (re-enqueue is theirs). |
| healthcheck: "running process uses ..., not the current release" | Someone switched `current` without a restart. Run `sudo /opt/crest-worker/bin/update.sh --force`. |
| Pins check FAIL | **Stop the worker** (`systemctl stop crest-worker`) and tell the user. Do not try to repair `events/` by hand. |

## 11. Decommission

1. `sudo systemctl disable --now crest-worker`. `ExecStopPost` releases its
   claims.
2. Confirm that `queue_status.py --own` reports no claim, and that
   `healthcheck.sh` shows no dead-process claims.
3. Remove the token: `sudo shred -u /etc/crest-worker/hf_token`. Revoke it on
   huggingface.co (Settings, then Access Tokens).
4. Commit `docs/JETSTREAM2_VALIDATION.md` from the kit checkout if it is not
   in git yet. Push only to fork `v2`.
5. In Exosphere: delete the instance, then any volume and the floating IP.
   Shelving only pauses the instance; deleting it stops all SU use.
6. If ZeroGPU alone is not enough afterwards, the user decides what replaces
   the worker (for example OSCER, per `OSCER_DISPATCH_BRIEF.md`).

## 12. Known gaps (worker code, not changed by this kit)

These need code changes on fork v2. They are the user's call:

- **No signal handling or checkpointing** in `crestimap/worker.py`. The kit
  compensates by stopping with SIGINT and releasing claims in
  `ExecStopPost`. A stop mid-visit still loses that visit's progress, and
  resident sessions are RAM-only. A persistent VM rarely stops, so the
  OSCER brief's `--max-hours`, `--idle-exit-min` and `--checkpoint-dir` are
  not required here.
- **Any exception consumes the job** as `failed.json`, including CUDA
  out-of-memory and `publish_event` returning False. The kit's VRAM ceiling
  check in the preflight reduces the risk.
- **No resolution check after the DEM loads.** `dem.load_dem` block-means
  when the merged raster exceeds `--max-cells`. The pre-claim `n_bbox` check
  prevents this only while the budget stays well above the queue cap (40 M
  versus 16 M today). Re-check this if `EVENT_QUEUE_MAX_CELLS` is raised.
  The G1 replay checks it.
- **Claim idents are `<hostname>:<pid>`** with no platform prefix. The
  dashboard labels only `hpc*` and `zerogpu*` engines, so it shows
  `jetstream2:<host>` raw.

## File map (`deploy/jetstream2/`, installed to `/opt/crest-worker/bin`)

| File | Role |
|---|---|
| `bootstrap.sh` | Idempotent setup; also `--set-token` and `--check-token`. Never starts the worker. |
| `update.sh` | New release, verify, wait for idle, stop, switch, start, roll back if it does not stay up; also `--check`, `--rollback`, `--force`, `--auto`. |
| `crest-worker-update.{service,timer}` | Daily `update.sh --auto` (hf_data refresh; off with `AUTO_UPDATE=0`). |
| `healthcheck.sh` | Read-only status (`--since 24h` for G4, `--quick`). |
| `crest-worker.service` | systemd unit: `Restart=always`, `LoadCredential` token, `KillSignal=SIGINT`, `TimeoutStopSec=180`, claim release on stop and start. |
| `crest-worker.env.example` | Becomes `/etc/crest-worker/crest-worker.env`: pins, budget, paths. No secrets. |
| `run-worker.sh` | ExecStart: pins the process to `current`, sets the engine label, execs `python -m crestimap.worker --sessions --publish`. |
| `preflight.sh` | ExecStartPre: waits for the GPU and the vGPU license; refuses unsafe config; clears RAM-orphaned session dirs. |
| `release_claims.py` | ExecStopPost/ExecStartPre: releases claims of dead processes of this host via `Worker._cleanup_queue(success=False)`. |
| `queue_status.py` | Read-only queue, claims, publishers, pins baseline/check, `--verify-publish`, `--whoami`, `--code-drift`. |
| `replay_bundle.py` | G1 offline replay (anonymous; cannot write). |
| `crest-py` | Runs those tools as `crest` with the env file, without the token. |
| `prune.sh` + `crest-worker-prune.{service,timer}` | Daily cache housekeeping. |
| `lib.sh` | Shared shell helpers (release build and verification). |
| `.gitattributes` | Keeps LF line endings in every checkout. |

VM layout:

| Path | Contents |
|---|---|
| `/opt/crest-worker/venv` | Python virtual environment. |
| `/opt/crest-worker/releases/<stamp>/{CREST-iMAP,CREST_AI,RELEASE.txt}` | Code releases. |
| `/opt/crest-worker/current`, `/opt/crest-worker/previous` | Symlinks to releases. |
| `/opt/crest-worker/bin` | Installed kit scripts. |
| `/etc/crest-worker/{crest-worker.env,hf_token}` | Configuration and the token. |
| `/var/lib/crest-worker` | The `crest` user's home: `huggingface.txt` (a link to the credential), `pins_baseline.json`, `updates.log`, `update.blocked` (only after a rolled-back update). |
| `/media/volume/crest-data/{jobs,dem_cache,hf_home,replay}` | Working data on the volume (`/var/lib/crest-worker/data/...` without one). |
