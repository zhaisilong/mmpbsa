# Local CPU and GPU scheduling

The local scheduler can overlap MD, trajectory processing, QC, and independent
GB/PB scoring without changing the scientific protocol. Physical CPU allocation
is opt-in: the historical defaults remain `execution_mode: legacy` and
`cpu_affinity: none`. A slot count alone does not guarantee disjoint CPU affinity.

## Physical resource leases

`cpu_affinity: physical` discovers Linux CPU topology within the calling process's
allowed affinity mask. It selects one logical CPU per physical core, distributes
the budget across NUMA nodes, and prefers a single NUMA node for each lease.
Insufficient allowed cores and unavailable affinity support fail explicitly.
External tools execute under `taskset`; invocation-local command runners preserve
the injectable backend interface without mutating a shared service or environment.
The Python worker's own native-thread affinity is restored before reuse.

Start the controller with `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1`, and `OMP_NUM_THREADS=1` **before importing numerical
libraries**. Existing library thread pools cannot be safely reconfigured per
worker. Set MD's command environment and `threads_per_job` separately to the
desired MD thread count and scoring MPI ranks separately.

The MPI launcher must honor the assigned CPU mask. The verified local OpenMPI
profile uses `--bind-to none`; do not inject OpenMPI options into unrelated MPI
implementations. GROMACS 2026.2 uses `mdrun_arguments: [-pin, inherit]`, rather
than overriding the external mask with `-pin on`. See the
[OpenMPI launcher reference](https://docs.open-mpi.org/en/v5.0.8/man-openmpi/man1/mpirun.1.html)
and [GROMACS thread-affinity reference](https://manual.gromacs.org/documentation/2026.2/user-guide/mdrun-performance.html).

`ResourceAllocation` and `StageRequest.allocation` carry the actual CPU/GPU IDs.
`CommandSpec.cpu_ids` passes CPU affinity into subprocess execution. These fields
are execution evidence, not science hash or backend compatibility inputs. Changing
the allocation does not invalidate a sealed PreparedCase or completed result.

## Stage pipeline

For a 32-physical-core budget and three MD GPUs, use:

```yaml
scheduler:
  execution_mode: pipeline
  cpu_affinity: physical
  cpu_slots: 32
  gpu_ids: ['5', '6', '7']  # example only; perform fresh GPU admission
  threads_per_job: 4
  max_workers: 13
  stage_limits: {md: 3, postprocess: 2, scoring: 8}
  cpu_reservations: {md: 12, postprocess: 2}
```

Pipeline `max_workers` limits executing **stages**, not whole jobs; legacy mode
retains its whole-job limit. `postprocess` shares one limit between trajectory
processing and trajectory QC. GB/PB count as separate scoring tasks.

One dispatcher owns the resource pool. It dispatches ready MD tasks first, then
postprocessing, then scoring, FIFO within each group. It acquires resources before
submitting work, so queued scoring tasks cannot occupy execution threads. CPU
reservations remain protected while their group has unfinished work, including
upstream-dependent tasks, and disappear once that group drains.

| Remaining work | Protected cores | Maximum four-rank scoring tasks |
| --- | --- | --- |
| MD and postprocessing | 12 MD plus 2 postprocessing | 4 |
| Postprocessing only | 2 postprocessing | 7 |
| Scoring only | None | 8 |

The stage graph, case-scoped admission, QC BLOCK, completion policy, declared model
order, and explicit run/resume/retry semantics are unchanged. A stage failure
releases its resources; independent jobs and scoring models continue. An error in
the dispatcher drains already-running stages before releasing their leases.

## Monitoring and recovery

The workspace's `scheduler_status.json` distinguishes queued tasks from executing
stages and lists their actual CPU/GPU IDs and terminal-stage counts. Each new
invocation's runtime record separates queue wait from actual execution time.
Historical `attempt.started_at` includes preparation and resource waiting; do not
use it alone to extrapolate scoring throughput.

Pipeline coordinators use an exclusive workspace lock. A controller must also
enforce its own campaign ownership and scientific gates. This is not Slurm or
OS-level isolation. Local mode requires projects to partition their budgets;
the opt-in cooperative registry below coordinates participating projects, but
never excludes unrelated users or tools from the same resources.

Before adopting this profile for a batch, verify live MPI rank affinity and
compare the same topology and frames in isolated single-task and concurrent runs.
Do not overwrite historical workflows, attempts, or source snapshots.

For a downstream geometry fix, `batch run --through scoring --new --resume-md`
is an explicit pipeline recovery operation. It reuses completed MD, resumes
abandoned active MD through the native checkpoint mechanism, and creates new
downstream attempts when science or compatibility changed. Matching MD science
and backend compatibility are mandatory; live owners, live subprocesses, and
changed MD science are rejected rather than silently restarted. A failed MD
attempt requires a provider's explicit native checkpoint retry hook: GROMACS
validates preceding phases and an available checkpoint/TPR, copies native state
into a fresh attempt with file hashes, and resumes there. The old failed attempt
remains unchanged. Missing checkpoints and unsupported providers fail explicitly;
no cold restart is substituted.
Stop the old coordinator and verify checkpoints before invoking this operation.
The default `--new` behavior is unchanged when `--resume-md` is absent.

`scheduler_status.json` additionally exposes `model_failures` and
`qc_blocked_jobs` while the batch is still running. QC BLOCK remains a scientific
exclusion, not a reason to loosen thresholds or score an invalid replica.

## Cooperative host resources (new projects)

Host coordination is opt-in. Missing `resource_coordination` preserves historical
process-local behavior. It requires Linux, pipeline, physical affinity, one UID,
host and PID namespace, and an explicit local registry file with an existing
parent directory. Network registries, cross-container sharing, MIG, remote MPI
and backends that daemonize and strip execution tokens are outside this first
cooperative implementation. It is not GPU-memory reservation or OS isolation.

Inspect topology/devices and explicitly choose a shared budget. These CPU IDs
and UUID are illustrative, not defaults:

```bash
mmpbsa resources init --registry /path/local/host-pool.json \
  --cpu 0,2,4,6 --gpu GPU-00000000-0000-0000-0000-000000000000
mmpbsa resources inspect --registry /path/local/host-pool.json
mmpbsa resources reconcile --registry /path/local/host-pool.json
# reconcile is dry-run unless --confirm is explicit; there is no --force
```

All participating projects use the same registry:

```yaml
scheduler:
  execution_mode: pipeline
  cpu_affinity: physical
  cpu_slots: 4               # project cap, not the host's entire CPU set
  threads_per_job: 2
  gpu_ids: ['0']             # checked against policy and normalized to UUID
  max_workers: 3
  resource_coordination:
    mode: host
    registry_path: /path/local/host-pool.json
```

CPU+GPU are granted atomically per stage. Physical package/core identities
include SMT aliases; allocation prefers NUMA-local representatives. GPU leases
use full UUID, child-local visibility and engine ordinal 0. Scheduler-owned
visibility/token do not enter scientific or OpenMM compatibility identity; user
visibility restrictions are not expanded. OpenMM host checkpoint resume also
requires matching recorded physical GPU UUIDs; legacy ordinals cannot prove it.

New grants use project rotation and project-local MD/postprocess/scoring
priority. Running work is non-preemptive, so an active long PB may still delay
another project. External GPU processes or unreliable observations cause waits,
never termination of others' tasks. Waiters hold no partial CPU/GPU lease or
execution thread. Polling is 5 seconds, heartbeat 30 seconds; waits warn after
15 minutes without an automatic timeout.

Leases bind boot ID, controller PID/start ticks, task/invocation and native
process/session evidence. Native descendants inherit a unique token, so changing
session does not release their resources. Living descendants after controller
exit are orphan-held; incomplete spawn or inaccessible process evidence is
unknown-held. Heartbeat expiry never proves resources free. Timeout/cancellation
only stops validated native processes of the current invocation. Corrupt
registries fail closed. Reboot requires a fresh explicitly initialized policy,
not silent reset. Inspect/dry-run reconcile do not create files or repair state.

## New project controller and exports

`project init` consumes sealed PreparedCase and a chosen runtime template,
refuses an existing destination and emits workflow/smoke/runtime/campaign YAML
plus a thin `run_project.py`. No chemistry is guessed. The launcher sets numerical
thread limits before imports; MD threads and scoring ranks remain explicit.

`project preflight` reads sealed inputs, dependency identities, actual CPUs,
host policy, stage-DAG dispatchability and free disk facts without creating a
workspace. Disk facts do not promise storage sufficiency for unknown trajectory
sizes. Smoke is an explicit native action in its own workspace, executing rep01
for all cases and all selected models. Production requires a passed gate bound
to prepared artifacts, settings, engine profile/overrides, source code and actual
dependency/environment identities. SDK project gates are trusted local callables
whose source identity is included. Changing inputs invalidates the gate; it does
not auto-resubmit smoke, and smoke does not replace production QC.

Explicit phases are smoke, production, finalize, report and export. Existing
job/batch resume/retry remain the recovery API; no implicit cold restart is added.
Report reuses assets and does not request an MD GPU. Offline export reads one
analysis and writes BOM CSVs, Chinese definitions and validation results; it
preserves source comparison groups/ranks and never merges project rankings.
Core requires neither RDKit, small-molecule SMILES, vendor IDs nor network
lookups. BLOCK scores and n=1 uncertainties remain blank, not zero.

`status --summary`/`project status` read without runner initialization, locks,
backend probes or resource discovery. Telemetry distinguishes internal queue,
shared/external wait and native execution. ETA requires comparable completed
samples and reports empirical conservative bounds, not confidence intervals.
Unknown dependency/external completion times remain unavailable; historical
attempt start timestamps are not native throughput evidence.
