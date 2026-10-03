# Isolated full-corpus training-memory measurements

All three planned processes completed successfully on NVIDIA GB10, driver
580.178.04, Linux aarch64, Python 3.14.8, Keras 3.15.1 and JAX/JAXLIB 0.11.2.
They completed exactly 10 AE epochs, 10 discriminator epochs and 830 GAN updates,
respectively. The verified controller phase window was 602.63 seconds, below
the declared 900-second aggregate cap; no process exceeded its 300-second cap.
Full configuration, package versions, immutable hashes, terminal counters,
phase-history endpoint snapshots and independent checks are retained in
the measurement evidence (local-only `../artifacts/roadmap-training-memory.json`).

| Selected workflow | External command wall time | End-of-phase host highwater | Whole-process peak RSS | End-of-phase device in-use peak | End-of-phase device pool peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| AE, 10 epochs | 251.24 s | 6.463 GiB | 6.493 GiB | 0.726 GiB | 1.002 GiB |
| Discriminator, 10 epochs | 258.76 s | 5.770 GiB | 5.824 GiB | 0.615 GiB | 1.002 GiB |
| GAN, 830 updates | 89.09 s | 5.781 GiB | 5.811 GiB | 0.710 GiB | 1.002 GiB |

Each workflow satisfied the local planning envelope of 16 GiB host memory and
4 GiB device allocator/pool highwater. The largest observed external process
RSS was 6.493 GiB; the largest sampled complete-family RSS was 6.559 GiB.
Post-export, pre-final-validation allocator in-use/pool peaks matched the
corresponding phase-end peaks. These observations support the declared envelope for this exact corpus,
batch size, dense model and recorded GB10 environment. They do not establish
minimum physical-memory requirements or guarantee the same peaks on other
hardware, runtime versions, models or concurrent workloads. No allocation
rewrite is justified by a budget breach in these runs.

This study measures the maintained dense training runner in a fresh GPU process
for each selected phase. It targets LOCAL-04's real-corpus phase-memory evidence.
It does not measure incremental kernel allocation, exact continuation, or
comparable model quality. The protocol was frozen on disk before launching any
of its GPU processes.

## Frozen inputs and budgets

All processes read the same existing `dataset-repeat/train_test.npz`: 106,248
training and 26,562 development-validation images, each 64×64 RGBA uint8. The
immutable archive SHA256 is
`1f97438e56695cac13898ac67f420c96f62ff3f3ba7df3ff7d49c21c3d290ac8`.
No source images, membership, normalization, or existing model artifacts are
rewritten. Data, code, lockfile, GPU requirements, benchmark scripts, and three
warm-start model hashes are checked before each launch and after the terminal
protocol state.

Each process calls the actual `run_training` implementation with dense
architecture, RGBA-MSE reconstruction, latent dimension 128, batch size 128,
seed 1976, bandwidth 3.16, learning rates 0.001/0.001/0.00001 for AE/D/G,
one discriminator update per GAN step, and checkpoint interval 50.

| Isolated workflow | AE epochs | Discriminator epochs | GAN steps | Initialization |
| --- | ---: | ---: | ---: | --- |
| Autoencoder | 10 | 0 | 0 | Cold seeded model initialization |
| Discriminator | 0 | 10 | 0 | Trusted `training-gpu-v2` encoder/decoder/discriminator weights |
| GAN | 0 | 0 | 830 | Same trusted weights, independently loaded |

Warm-start deliberately resets optimizer state, progress, and RNG. The
warm-started weights come from a previous completed full training run, including
its GAN phase. Consequently these isolated runs are memory probes, not a
reconstruction of the original sequential training trajectory or an AE-only
versus GAN quality comparison.

Processes run sequentially, each with a 300-second timeout and a 900-second
aggregate budget. The planning limits are 16 GiB host memory and 4 GiB device
allocation/pool highwater. The controller stops on a failed process, timeout,
observed memory-limit violation, or changed frozen input. It retains partial
artifacts and does not retry. Host family RSS is sampled every two seconds;
shared pages can be counted multiple times in that conservative sum. Device
limits are checked against published checkpoint allocator highwaters, not
continuous CUDA VRAM samples. Thus these are observed-budget checks, not a
hard operating-system memory reservation or per-allocation device guard.

An explicit monitoring amendment is retained with the local protocol: the
original controller's main-thread child traversal missed Python descendants
spawned by uv worker threads. A supplemental guard discovers the complete PPid
ancestry and samples it every second. It was attached during the AE process;
the first portion of that phase therefore has incomplete family-RSS sampling.
The frozen original controller was retained unchanged and no phase was restarted.
External `/usr/bin/time` remains the independent whole-process peak measurement,
including that early portion. This limitation affects live monitoring rather
than the terminal peak-RSS evidence or device checkpoint counters.

## What the measurements include

A selected-phase history snapshot is taken by the runner after the final phase
update and before model export and final validation. It reports process-lifetime
host RSS highwater and JAX allocator counters in a fresh process, eliminating
JIT/caches from earlier training phases in that same process. It still includes
imports, archive decompression, full model initialization, and checkpointing.
The phase-end snapshot precedes that phase's final checkpoint publication;
earlier checkpoints are included in the lifetime counter. External whole-process
RSS also includes the final checkpoint.
The AE history snapshot precedes encoder-code extraction, which still belongs
to that workflow's externally measured whole-process RSS and elapsed time.
Discriminator/GAN snapshots additionally include warm-start loading, encoder
code extraction, KDE preparation, and decoder/discriminator setup. They are
isolated phase workflows, not the marginal cost of a single optimizer update.

External `/usr/bin/time -v` peak RSS includes the entire successful process,
including final model/bundle export and validation. The runner's later allocator
snapshot follows export but precedes final validation; the device peak after
that validation is not independently sampled. It is reported separately from
the selected-phase snapshot. Allocated bytes and
pool highwater differ; neither is a complete measurement of system GPU use.
The GPU's reported total memory may be unavailable and is not inferred from
JAX's allocation limit.

GPU settings retain the pinned optional CUDA 13 profile with
`JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MPLBACKEND=Agg`,
`OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Complete runtime package
versions, hardware/driver details, immutable input fingerprints, actual phase
histories and counters, timings, and terminal outcomes are retained with the
experiment evidence. The workload and memory limits are specific to the
recorded hardware/runtime and do not establish portable minimum requirements.

Independent post-run checks rehashed every frozen source/data/weight/lock input,
parsed the NPZ's actual array headers, compared recorded configuration and
warm-start hashes, and verified all expected progress counters and history
indices. Every recorded AE validation metric, discriminator epoch aggregate and
GAN per-update loss is finite. A separate CPU scan checked all saved numeric
model/optimizer datasets across the twelve endpoint model files. AE/D per-batch
losses and intermediate gradients are not retained by this runner, so this
evidence does not claim to inspect those unrecorded values. Post-run CPU checks
are separate processes and are excluded from the training-resource measurements.

## Local reproduction

The frozen `protocol.json`, controller `launch.py`, phase wrapper `run_phase.py`,
logs, resource reports, and new isolated run directories live under
`images/results/roadmap-2026-10-02/isolated-training-memory/`.
This local study depends on the prepared full corpus and trusted prior weights;
it is not a fresh-clone data or training guarantee. The wrapper uses the locked
uv environment and exact `requirements-gpu.txt`; it calls the maintained runner
without callbacks, monkeypatches, or changing the working directory.

To repeat, create a new experiment directory and new run/output paths, freeze a
new protocol with the same declared hashes and budgets, then execute its
controller from the project root:

```sh
python3 images/results/roadmap-2026-10-02/isolated-training-memory/launch.py
```

The existing directory is a retained observation and must not be reused: both
logs and maintained run directories refuse collisions. Results from a new
protocol must identify its own fingerprints and terminal outcomes.
