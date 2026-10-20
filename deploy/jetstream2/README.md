# Jetstream2 GPU worker kit

This kit sets up a persistent Jetstream2 GPU VM (g3.large recommended) as a
CREST-iMAP v2 queue worker. The VM runs the existing headless worker,
`python -m crestimap.worker --sessions --publish`, under systemd.

The full runbook is in [`docs/JETSTREAM2_WORKER.md`](../../docs/JETSTREAM2_WORKER.md).
It covers creating the VM, the token, validation gates G0–G4, day-2
operations, the HPC handoff and decommissioning.

Quick start, from an SSH shell on the VM:

```bash
# data on the attached 300 GB volume (runbook section 2); idempotent; never starts the worker
sudo INIT_DATA_ROOT=/media/volume/crest-data bash ~/crest-kit/deploy/jetstream2/bootstrap.sh
sudo /opt/crest-worker/bin/bootstrap.sh --set-token         # PI: fine-grained HF token, input hidden
sudo /opt/crest-worker/bin/healthcheck.sh                   # read-only status
# then gates G0 and G1, and only with the user's go-ahead G2:
sudo systemctl enable --now crest-worker
```

bootstrap also enables a daily timer (`crest-worker-update.timer`) that keeps
the worker's `hf_data` on CREST_AI `main`, restarting only between visits.

Nothing in this folder is imported by any Space or by the `crestimap`
package, so committing or pushing it changes no running service.
