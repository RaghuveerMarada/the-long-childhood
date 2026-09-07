# =============================================================================
# scripts/ml/cloud/launch.py
#
# Local-side launcher: stages the repo to GCS, then spins up one or more
# Compute Engine VMs to run jobs from the manifest.
#
# Usage:
#   python3 scripts/ml/cloud/launch.py stage          # tar+upload repo
#   python3 scripts/ml/cloud/launch.py smoke          # one tiny VM
#   python3 scripts/ml/cloud/launch.py panel          # spin up the full run
#   python3 scripts/ml/cloud/launch.py status         # show running VMs
#   python3 scripts/ml/cloud/launch.py cleanup        # delete all chapter9 VMs
# =============================================================================
"""
launch.py — local-side orchestrator for the Panel ML jobs on GCP.

Conservative defaults: preemptible T4 instances in us-central1-a; CPU
fallback for non-GPU jobs; auto-shutdown on job completion. All logs +
artifacts stream to the bucket named by $GCS_BUCKET.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time

# GCP target — read from the environment so no project/bucket id is
# committed to the repo. Set before running, e.g.:
#   export GCP_PROJECT=your-gcp-project-id
#   export GCS_BUCKET=your-results-bucket
PROJECT = os.environ.get("GCP_PROJECT", "")
BUCKET = os.environ.get("GCS_BUCKET", "")
REGION = "us-central1"
ZONE = "us-central1-a"
VM_PREFIX = "ch9"
WORKER_PREFIX = "ch9w"   # persistent worker-pool VMs (distinct from per-job ch9-)

# One T4 zone per region that has 4x preemptible-T4 quota (verified via
# `gcloud compute accelerator-types list`). 19 regions x 4 = up to 76
# concurrent workers. Spreading this wide also dilutes preemption/stockout
# risk: a capacity crunch in one region barely dents the fleet.
ZONES_ALL = [
    "us-central1-a", "us-east1-b", "us-east4-a", "us-west1-a",
    "us-west2-b", "us-west3-b", "us-west4-a",
    "northamerica-northeast1-c", "southamerica-east1-a",
    "europe-west1-b", "europe-west2-a", "europe-west3-b", "europe-west4-a",
    "asia-east1-a", "asia-northeast1-a", "asia-northeast3-b",
    "asia-south1-a", "asia-southeast1-a", "australia-southeast1-a",
]

# Deep Learning VM with PyTorch + CUDA preinstalled
GPU_IMAGE_FAMILY = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
GPU_IMAGE_PROJECT = "deeplearning-platform-release"
GPU_MACHINE = "n1-standard-4"      # cheapest GPU-compatible
GPU_ACCEL = "type=nvidia-tesla-t4,count=1"

CPU_IMAGE_FAMILY = "debian-12"
CPU_IMAGE_PROJECT = "debian-cloud"
CPU_MACHINE = "n2-standard-8"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTUP_SCRIPT = os.path.join(SCRIPT_DIR, "startup.sh")


def run(cmd, check=True, capture=False, **kw):
    print(f"$ {' '.join(cmd)}")
    if capture:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, **kw)
    return subprocess.run(cmd, check=check, **kw)


def ensure_bucket():
    """Make the bucket if missing."""
    r = subprocess.run(["gsutil", "ls", f"gs://{BUCKET}"],
                        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Creating bucket gs://{BUCKET} ...")
        run(["gsutil", "mb", "-p", PROJECT, "-l", REGION, f"gs://{BUCKET}"])
    else:
        print(f"Bucket gs://{BUCKET} exists")


def stage_repo():
    """Build a minimal tarball of the repo (scripts/, data/, wcde/data/processed/)
    and upload it to GCS. Returns the GCS URI."""
    ensure_bucket()
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = f"repo_snapshot_{timestamp}.tar.gz"
    tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    tmp.close()

    # Tarball includes what the ML job needs to run. Excludes paper/,
    # book/, essay/, anything that would bloat the upload.
    includes = [
        "scripts/",
        "wcde/data/processed/",
        "data/",
        "Makefile",
    ]
    excludes_re = [
        ".git/", ".pytest_cache/", "__pycache__/", ".DS_Store",
        "checkpoints/", "review/", "essay/",
        "book/", "book_serious/", "paper/", "vision/",
        "checkin/website/", "*.pdf", "*.aux", "*.log",
    ]
    with tarfile.open(tmp.name, "w:gz") as t:
        for inc in includes:
            src = os.path.join(REPO_ROOT, inc)
            if not os.path.exists(src):
                print(f"  skip (missing): {inc}")
                continue
            for root, dirs, files in os.walk(src):
                # prune
                dirs[:] = [d for d in dirs if not any(d.startswith(ex.rstrip("/"))
                                                      for ex in excludes_re)]
                for fn in files:
                    full = os.path.join(root, fn)
                    if any(ex in full for ex in excludes_re):
                        continue
                    arcname = os.path.join("repo", os.path.relpath(full, REPO_ROOT))
                    t.add(full, arcname=arcname)

    size_mb = os.path.getsize(tmp.name) / 1e6
    print(f"Tarball: {tmp.name}  ({size_mb:.1f} MB)")
    gcs_uri = f"gs://{BUCKET}/code/{name}"
    run(["gsutil", "-q", "cp", tmp.name, gcs_uri])
    os.unlink(tmp.name)
    print(f"Staged: {gcs_uri}")
    # Also write a "latest" pointer
    run(["gsutil", "-q", "cp", "-r", gcs_uri, f"gs://{BUCKET}/code/latest.tar.gz"])
    return gcs_uri


def latest_code_uri():
    return f"gs://{BUCKET}/code/latest.tar.gz"


def launch_vm(job_id, job_kind, job_params, gpu=True, preempt=True,
              shutdown=True, dry_run=False, zone=ZONE):
    """Spin up a VM to run one job in `zone`."""
    vm_name = f"{VM_PREFIX}-{job_id.lower().replace('_', '-')}"
    vm_name = vm_name[:62]  # GCE limit
    gcs_out = f"gs://{BUCKET}/results/{job_id}/"

    metadata_kv = [
        ("job-id", job_id),
        ("job-kind", job_kind),
        ("job-params", json.dumps(job_params)),
        ("gcs-code-uri", latest_code_uri()),
        ("gcs-output-uri", gcs_out),
        ("shutdown-on-completion", "true" if shutdown else "false"),
        # Propagate the generation-unification flag from the launching shell
        # into the VM so the remote data_loader trains at LAG_GENERATION (28)
        # rather than its local default (legacy 25 grid). Without this the env
        # var never crosses the boundary and the VM silently trains at 25.
        ("ml-unify-generation", os.environ.get("ML_UNIFY_GENERATION", "0")),
    ]
    if gpu:
        metadata_kv.append(("install-nvidia-driver", "True"))

    # Use --metadata-from-file for everything (startup-script + each key/val
    # written to a temp file). This avoids gcloud's quoting issues with JSON
    # values that contain commas inside --metadata.
    tmp_meta_dir = tempfile.mkdtemp(prefix=f"meta_{job_id}_")
    meta_file_args = [f"startup-script={STARTUP_SCRIPT}"]
    for k, v in metadata_kv:
        fp = os.path.join(tmp_meta_dir, k)
        with open(fp, "w") as f:
            f.write(v)
        meta_file_args.append(f"{k}={fp}")
    metadata_from_file = ",".join(meta_file_args)

    cmd = [
        "gcloud", "compute", "instances", "create", vm_name,
        f"--project={PROJECT}",
        f"--zone={zone}",
        f"--metadata-from-file={metadata_from_file}",
        "--scopes=cloud-platform",
        "--boot-disk-size=100GB",
    ]
    if gpu:
        cmd += [
            f"--machine-type={GPU_MACHINE}",
            f"--image-family={GPU_IMAGE_FAMILY}",
            f"--image-project={GPU_IMAGE_PROJECT}",
            f"--accelerator={GPU_ACCEL}",
            "--maintenance-policy=TERMINATE",
        ]
    else:
        cmd += [
            f"--machine-type={CPU_MACHINE}",
            f"--image-family={CPU_IMAGE_FAMILY}",
            f"--image-project={CPU_IMAGE_PROJECT}",
        ]
    if preempt:
        cmd += ["--preemptible", "--no-restart-on-failure"]

    if dry_run:
        print(f"DRY RUN: would launch {vm_name}")
        print("  " + " ".join(cmd))
        return vm_name

    try:
        run(cmd, check=True, capture=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        msg = f"  ! launch failed for {vm_name}: {stderr.strip()[:200]}"
        print(msg)
        return None
    print(f"Launched: {vm_name}  →  {gcs_out}")
    return vm_name


def list_running():
    r = run(["gcloud", "compute", "instances", "list",
             f"--project={PROJECT}",
             f"--filter=name~^{VM_PREFIX}-",
             "--format=value(name,status,zone,scheduling.preemptible)"],
            capture=True)
    print(r.stdout)


def cleanup_all(dry_run=False):
    """Delete every chapter9 VM — both per-job (ch9-) and worker-pool (ch9w-)."""
    r = run(["gcloud", "compute", "instances", "list",
             f"--project={PROJECT}",
             f"--filter=name~^{VM_PREFIX}",
             "--format=value(name,zone)"],
            capture=True)
    if not r.stdout.strip():
        print("No chapter9 VMs to delete")
        return
    for line in r.stdout.strip().splitlines():
        name, zone = line.split("\t")
        zone = zone.split("/")[-1]
        if dry_run:
            print(f"DRY RUN: would delete {name} in {zone}")
        else:
            run(["gcloud", "compute", "instances", "delete", name,
                 f"--zone={zone}", f"--project={PROJECT}", "--quiet"])


def cmd_smoke(args):
    stage_repo()
    vm = launch_vm(
        job_id=f"smoke-{int(time.time())}",
        job_kind="smoke", job_params={},
        gpu=args.gpu, preempt=True, shutdown=True, dry_run=args.dry_run,
    )
    print(f"\nSmoke job launched: {vm}")
    print(f"  Watch logs:    gcloud compute ssh {vm} --zone={ZONE} -- 'tail -f /var/log/job.log'")
    print(f"  Watch results: gsutil ls gs://{BUCKET}/results/")
    print(f"  Estimated runtime: ~5-10 minutes (boot + driver install + 1 ridge fit)")
    print(f"  Estimated cost: < $0.50")


def cmd_panel(args):
    """Launch the full Panel job set respecting a concurrency cap.

    Strategy: stage code once, then launch up to --max-concurrent VMs at a
    time. Each VM is self-deleting on completion. New VMs are launched as
    old ones finish (detected by result.json appearing in GCS).
    """
    import job_manifest as jm
    jobs = jm.panel_jobs()
    summary = jm.manifest_summary(jobs)
    cost_spot = summary.get("est_cost_usd_spot",
                              summary.get("est_cost_usd_preempt", 0.0))
    print(f"Big run: {summary['n_jobs']} jobs, "
          f"est {summary['total_gpu_hours']:.1f} GPU-hr + "
          f"{summary['total_cpu_hours']:.1f} CPU-hr, "
          f"est cost spot ${cost_spot:.2f}")

    if args.only:
        wanted = set(args.only.split(","))
        jobs = [j for j in jobs if j["id"] in wanted or j["kind"] in wanted]
        print(f"Filtered to {len(jobs)} jobs")

    if not args.skip_stage:
        stage_repo()
    else:
        print("Skipping repo stage (--skip-stage)")

    if args.dry_run:
        print("\nDRY RUN — would launch:")
        for j in jobs:
            print(f"  {j['id']:<40} {j['kind']:<25} "
                  f"gpu={j['est_gpu_hours']:.2f} cpu={j['est_cpu_hours']:.2f}")
        return

    # Parse zone list for round-robin VM placement
    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    if not zones:
        zones = [ZONE]
    zone_cap = max(1, args.max_concurrent // len(zones))
    print(f"Zones: {zones}  zone_cap={zone_cap}  "
          f"total_cap={args.max_concurrent}")

    # Concurrency loop with preempt-recovery
    in_flight = {}     # vm_name -> {"job": j, "started": ts, "retries": k, "zone": z}
    job_by_id = {j["id"]: j for j in jobs}
    pending = list(jobs)
    completed = set(_completed_jobs_in_gcs())
    pending = [j for j in pending if j["id"] not in completed]
    if completed:
        print(f"Resuming: {len(completed)} jobs already complete in GCS, "
              f"{len(pending)} remaining")
    MAX_RETRIES = 3
    retry_counts = {j["id"]: 0 for j in jobs}

    # Adopt any VMs that are already running from a previous orchestrator
    existing = _existing_vms_to_job_ids()
    for vm, (jid, vm_zone) in existing.items():
        if jid in job_by_id:
            in_flight[vm] = {"job": job_by_id[jid], "started": time.time(),
                              "retries": 0, "zone": vm_zone}
            print(f"  ↻ adopting existing VM {vm} in {vm_zone} (job {jid})")
            # Remove from pending so we don't relaunch
            pending = [j for j in pending if j["id"] != jid]

    zone_rr_idx = 0

    def pick_zone_with_capacity():
        """Round-robin zone with available slot; None if all zones full."""
        nonlocal zone_rr_idx
        n = len(zones)
        for offset in range(n):
            idx = (zone_rr_idx + offset) % n
            z = zones[idx]
            n_here = sum(1 for m in in_flight.values()
                          if m.get("zone") == z)
            if n_here < zone_cap:
                zone_rr_idx = (idx + 1) % n
                return z
        return None

    while pending or in_flight:
        # Launch up to max-concurrent, respecting per-zone caps
        while pending and len(in_flight) < args.max_concurrent:
            z = pick_zone_with_capacity()
            if z is None:
                break  # all zones at cap; wait for completions
            j = pending.pop(0)
            gpu = j["est_gpu_hours"] > 0
            vm = launch_vm(
                job_id=j["id"], job_kind=j["kind"], job_params=j["params"],
                gpu=gpu, preempt=True, shutdown=True, dry_run=False,
                zone=z,
            )
            if vm is None:
                # Launch failed; back of queue for retry, sleep a beat
                retry_counts[j["id"]] += 1
                if retry_counts[j["id"]] <= MAX_RETRIES:
                    pending.append(j)
                else:
                    print(f"  ✗ {j['id']} skipped after {MAX_RETRIES} "
                          f"failed launch attempts")
                time.sleep(10)
                continue
            in_flight[vm] = {"job": j, "started": time.time(),
                              "retries": retry_counts[j["id"]], "zone": z}
            print(f"  in_flight={len(in_flight)} pending={len(pending)} "
                  f"queued={vm} ({z})", flush=True)

        if not in_flight:
            break

        # Wait, then poll completions + check for preempted/dead VMs
        time.sleep(90)
        done = _completed_jobs_in_gcs()
        alive = _list_running_vms()

        for vm, meta in list(in_flight.items()):
            jid = meta["job"]["id"]
            if jid in done:
                print(f"  ✓ {jid}")
                del in_flight[vm]
                continue
            # If VM is gone from instance list but job not in done set,
            # it was preempted (or self-deleted with no result, which is
            # an error). Re-queue if under retry limit.
            if vm not in alive:
                meta["retries"] += 1
                retry_counts[jid] = meta["retries"]
                if meta["retries"] <= MAX_RETRIES:
                    print(f"  ⟲ {jid} preempted in {meta['zone']}; "
                          f"re-queue (retry {meta['retries']}/{MAX_RETRIES})")
                    pending.append(meta["job"])
                else:
                    print(f"  ✗ {jid} failed after {MAX_RETRIES} retries; "
                          f"giving up")
                del in_flight[vm]
                continue
            # Watchdog: if a VM has been alive > 2 hours, something is wrong
            if time.time() - meta["started"] > 2 * 3600:
                print(f"  ⏰ {jid} > 2h on VM {vm} ({meta['zone']}); "
                      f"killing and re-queueing")
                _delete_vm(vm, meta["zone"])
                meta["retries"] += 1
                retry_counts[jid] = meta["retries"]
                if meta["retries"] <= MAX_RETRIES:
                    pending.append(meta["job"])
                del in_flight[vm]

    print(f"\nPhase A: complete. {len(completed)} (initial) + "
          f"{len(jobs) - len(pending) - len(in_flight) - len(completed)} new = "
          f"{len(_completed_jobs_in_gcs())} jobs in GCS.")


WORKER_STARTUP = os.path.join(SCRIPT_DIR, "startup_worker.sh")


def launch_worker_vm(vm_name, zone, worker_id, dry_run=False):
    """Spin up a persistent worker VM that loops over the GCS claim queue."""
    metadata_kv = [
        ("gcs-code-uri", latest_code_uri()),
        ("bucket", BUCKET),
        ("worker-id", worker_id),
        ("install-nvidia-driver", "True"),
        # See launch_vm: carry ML_UNIFY_GENERATION into the worker so its jobs
        # train at LAG_GENERATION (28) instead of the legacy 25-year grid.
        ("ml-unify-generation", os.environ.get("ML_UNIFY_GENERATION", "0")),
    ]
    tmp_meta_dir = tempfile.mkdtemp(prefix=f"meta_{worker_id}_")
    meta_file_args = [f"startup-script={WORKER_STARTUP}"]
    for k, v in metadata_kv:
        fp = os.path.join(tmp_meta_dir, k)
        with open(fp, "w") as f:
            f.write(v)
        meta_file_args.append(f"{k}={fp}")
    cmd = [
        "gcloud", "compute", "instances", "create", vm_name,
        f"--project={PROJECT}", f"--zone={zone}",
        f"--metadata-from-file={','.join(meta_file_args)}",
        "--scopes=cloud-platform", "--boot-disk-size=100GB",
        f"--machine-type={GPU_MACHINE}",
        f"--image-family={GPU_IMAGE_FAMILY}",
        f"--image-project={GPU_IMAGE_PROJECT}",
        f"--accelerator={GPU_ACCEL}",
        "--maintenance-policy=TERMINATE",
        "--preemptible", "--no-restart-on-failure",
    ]
    if dry_run:
        print(f"DRY RUN: would launch worker {vm_name} in {zone}")
        return vm_name
    try:
        run(cmd, check=True, capture=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        print(f"  ! worker launch failed {vm_name} ({zone}): {stderr.strip()[:160]}")
        return None
    print(f"Launched worker: {vm_name}  ({zone}, {worker_id})")
    return vm_name


def _list_worker_vms():
    """Return {worker_vm_name: zone} for all live worker-pool VMs."""
    r = subprocess.run(
        ["gcloud", "compute", "instances", "list",
         f"--project={PROJECT}", f"--filter=name~^{WORKER_PREFIX}-",
         "--format=value(name,zone)"],
        capture_output=True, text=True,
    )
    out = {}
    if r.returncode != 0:
        return out
    for line in r.stdout.splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1].split("/")[-1]
    return out


def cmd_worker_pool(args):
    """Persistent worker pool: build a GCS claim queue, launch N workers that
    each boot once and run many jobs, and monitor the fleet — topping it up
    after preemptions and reaping claims left by dead workers — until every
    job has a result.json in GCS."""
    import job_manifest as jm
    import queue_lib as q

    jobs = jm.panel_jobs()
    if getattr(args, "only", ""):
        prefixes = [p.strip() for p in args.only.split(",") if p.strip()]
        def _match(j):
            return any(j["id"].startswith(p) or j["kind"].startswith(p)
                       for p in prefixes)
        before = len(jobs)
        jobs = [j for j in jobs if _match(j)]
        print(f"--only={args.only}: filtered {before} -> {len(jobs)} jobs")
    summary = jm.manifest_summary(jobs)
    print(f"Worker pool: {summary['n_jobs']} jobs, "
          f"est {summary['total_gpu_hours']:.1f} GPU-hr + "
          f"{summary['total_cpu_hours']:.1f} CPU-hr")

    if not args.skip_stage:
        stage_repo()
    else:
        print("Skipping repo stage (--skip-stage)")

    n_built = q.build_queue(BUCKET, jobs)
    print(f"Queue built: {n_built} job specs -> gs://{BUCKET}/queue/manifest.json")

    zones = [z.strip() for z in args.zones.split(",") if z.strip()] or ZONES_ALL
    target = args.workers
    job_ids = {j["id"] for j in jobs}

    if args.dry_run:
        print(f"DRY RUN — would launch {target} workers across {len(zones)} zones:")
        for i in range(target):
            print(f"  worker w{i:03d} -> {zones[i % len(zones)]}")
        return

    # Adopt any workers already running (orchestrator restart).
    workers = _list_worker_vms()
    if workers:
        print(f"Adopting {len(workers)} existing workers")
    next_idx = len(workers)

    def zone_for(idx):
        return zones[idx % len(zones)]

    POLL = 90
    while True:
        done = set(_completed_jobs_in_gcs()) & job_ids
        if len(done) >= len(job_ids):
            print(f"\nAll {len(job_ids)} jobs complete.")
            break

        workers = _list_worker_vms()
        # Reap claims left by workers that no longer exist.
        reaped = q.reap_stale_claims(BUCKET, set(workers.keys()))
        if reaped:
            print(f"  reaped {len(reaped)} stale claim(s): "
                  f"{', '.join(reaped[:5])}{'...' if len(reaped) > 5 else ''}")

        # Top the fleet back up to target (only as many as remaining work).
        # Launch in PARALLEL — serial gcloud create is ~17s each, so ramping
        # to 40+ workers serially took ~10 min; firing them concurrently
        # ramps the whole fleet in ~one create's worth of wall-clock.
        remaining = len(job_ids) - len(done)
        want = min(target, remaining)
        need = want - len(workers)
        if need > 0:
            batch = []
            for _ in range(need):
                zone = zone_for(next_idx)
                wid = f"w{next_idx:03d}"
                vm = f"{WORKER_PREFIX}-{zone}-{wid}"[:62]
                batch.append((vm, zone, wid))
                next_idx += 1
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(32, len(batch))) as ex:
                futs = {ex.submit(launch_worker_vm, vm, zone, wid): vm
                        for (vm, zone, wid) in batch}
                for fut in futs:
                    vm = futs[fut]
                    try:
                        res = fut.result()
                    except Exception:
                        res = None
                    if res is not None:
                        zone = next(z for (v, z, _) in batch if v == vm)
                        workers[vm] = zone

        print(f"  workers={len(workers)} done={len(done)}/{len(job_ids)} "
              f"remaining={remaining}", flush=True)
        time.sleep(POLL)

    # Drained: clean up any lingering workers (they self-delete, but be sure).
    leftover = _list_worker_vms()
    for vm, zone in leftover.items():
        _delete_vm_in_zone(vm, zone)
    print("Worker pool: complete.")


def _delete_vm_in_zone(vm_name, zone):
    subprocess.run(
        ["gcloud", "compute", "instances", "delete", vm_name,
         f"--zone={zone}", f"--project={PROJECT}", "--quiet"],
        capture_output=True,
    )


def _list_running_vms():
    """Return set of currently-running ch9-* VM names."""
    r = subprocess.run(
        ["gcloud", "compute", "instances", "list",
         f"--project={PROJECT}",
         f"--filter=name~^{VM_PREFIX}-",
         "--format=value(name)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return set()
    return set(line.strip() for line in r.stdout.splitlines() if line.strip())


def _list_running_vms_with_zone():
    """Return {vm_name: zone} for all ch9-* VMs across all zones."""
    r = subprocess.run(
        ["gcloud", "compute", "instances", "list",
         f"--project={PROJECT}",
         f"--filter=name~^{VM_PREFIX}-",
         "--format=value(name,zone)"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return {}
    out = {}
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        zone = parts[1].split("/")[-1]
        out[name] = zone
    return out


def _existing_vms_to_job_ids():
    """Map existing ch9-* VM names to (job_id, zone) tuples.

    Used at orchestrator restart: VMs running from a previous orchestrator
    instance must be treated as in-flight so we don't try to relaunch them.
    Zone is captured so the new orchestrator's watchdog can kill adopted
    VMs in their actual zone, even if they were launched in a different one.
    """
    vms = _list_running_vms_with_zone()
    out = {}
    for vm_name, zone in vms.items():
        if not vm_name.startswith(f"{VM_PREFIX}-"):
            continue
        suffix = vm_name[len(VM_PREFIX) + 1:]
        job_id = suffix.replace("-", "_")
        out[vm_name] = (job_id, zone)
    return out


def _delete_vm(vm_name, zone):
    subprocess.run(
        ["gcloud", "compute", "instances", "delete", vm_name,
         f"--zone={zone}", f"--project={PROJECT}", "--quiet"],
        capture_output=True,
    )


def _completed_jobs_in_gcs():
    """Return set of job IDs that have result.json in their GCS path.

    Uses a single recursive ls to avoid N+1 gsutil round-trips. With ~300
    jobs the old per-job stat loop took 10+ minutes; this version is one
    call.
    """
    r = subprocess.run(
        ["gsutil", "-q", "ls", f"gs://{BUCKET}/results/**/result.json"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return set()
    completed = set()
    prefix = f"gs://{BUCKET}/results/"
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith(prefix) or not line.endswith("/result.json"):
            continue
        rel = line[len(prefix):-len("/result.json")]
        if rel:
            completed.add(rel)
    return completed


def cmd_aggregate(args):
    """Pull all results from GCS into a local directory and merge into one JSON."""
    out_dir = os.path.join(REPO_ROOT, "scripts", "ml", "checkin", "panel_runs")
    os.makedirs(out_dir, exist_ok=True)
    run(["gsutil", "-m", "-q", "rsync", "-r",
         f"gs://{BUCKET}/results/", out_dir])
    print(f"Pulled all results to {out_dir}")

    # Build merged JSON
    merged = {}
    for jd in sorted(os.listdir(out_dir)):
        rj = os.path.join(out_dir, jd, "result.json")
        if not os.path.exists(rj):
            continue
        with open(rj) as f:
            merged[jd] = json.load(f)
    with open(os.path.join(out_dir, "_merged.json"), "w") as f:
        json.dump(merged, f, indent=2, default=float)
    print(f"Merged: {len(merged)} jobs → _merged.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sp = sub.add_parser("stage"); sp.set_defaults(func=lambda a: stage_repo())
    sp = sub.add_parser("smoke"); sp.add_argument("--gpu", action="store_true",
                                                    default=True)
    sp.add_argument("--no-gpu", dest="gpu", action="store_false")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_smoke)
    sp = sub.add_parser("status"); sp.set_defaults(func=lambda a: list_running())
    sp = sub.add_parser("cleanup"); sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=lambda a: cleanup_all(dry_run=a.dry_run))
    sp = sub.add_parser("panel")
    sp.add_argument("--max-concurrent", type=int, default=4,
                     help="Max concurrent VMs across all zones (T4 quota is per-region)")
    sp.add_argument("--zones", default=ZONE,
                     help="Comma-separated zones to round-robin VMs across. "
                          "T4 quota is per-region so use one zone per region. "
                          "Default: us-central1-a only.")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--skip-stage", action="store_true",
                     help="Skip repo staging (use already-uploaded latest.tar.gz)")
    sp.add_argument("--only", default="",
                     help="Comma-separated job IDs or kinds to filter to")
    sp.set_defaults(func=cmd_panel)
    sp = sub.add_parser("worker-pool")
    sp.add_argument("--workers", type=int, default=40,
                     help="Target number of persistent workers (capped by "
                          "per-region T4 quota across the zone set)")
    sp.add_argument("--zones", default=",".join(ZONES_ALL),
                     help="Comma-separated T4 zones to spread workers across. "
                          "Default: all 19 regions with 4x preemptible-T4 quota.")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--skip-stage", action="store_true",
                     help="Skip repo staging (use already-uploaded latest.tar.gz)")
    sp.add_argument("--only", default="",
                     help="Comma-separated job-ID or kind prefixes to filter the "
                          "queue to. Example: --only longrun_ runs only the 20 "
                          "longrun jobs.")
    sp.set_defaults(func=cmd_worker_pool)
    sp = sub.add_parser("aggregate"); sp.set_defaults(func=cmd_aggregate)
    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        sys.exit(1)
    if not PROJECT or not BUCKET:
        sys.exit("error: set GCP_PROJECT and GCS_BUCKET (your GCP project id "
                 "and results bucket name) before running launch.py")
    args.func(args)


if __name__ == "__main__":
    main()
