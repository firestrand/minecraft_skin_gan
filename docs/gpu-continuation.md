# GPU checkpoint continuation: measured results and limits

The default GPU profile failed the unchanged strict continuation comparison.
A separate study using explicit deterministic XLA startup flags passed all three
interruption phases on the recorded NVIDIA GB10 environment. The default failure
is retained alongside the successful study; it was not overwritten, omitted,
or made to pass by weakening tolerances.

| Study | Fresh processes | Interrupted phases compared | Result |
| --- | ---: | --- | --- |
| Default GPU compiler profile | 3 | AE; study stopped at the first failed comparison | Best-validation difference ≈1.385193×10⁻⁶ exceeded the original ≈1.925481×10⁻⁷ allclose threshold |
| Explicit deterministic startup profile | 7 | AE, discriminator, GAN | Every final checkpoint array matched exactly; maximum absolute array error 0 in each comparison |

For the deterministic study, each resumed final checkpoint contained the same
84 saved model, selected-best-AE, and optimizer arrays as the uninterrupted
reference. The checkpoint NPZ files also had identical byte hashes. RNG state,
completed counters, numeric training histories, stored encoder codes/bandwidth,
and final validation MSE matched exactly. Final validation MSE was
`0.09165017527788709` for the reference and all three resumed runs. Original
array/metric tolerances remained `rtol=1e-6`, `atol=1e-7`; RNG and counters required
exact equality. Timing and memory observations were excluded from numeric
training-history equality.

Evidence is retained in the default-profile report (local-only `../artifacts/roadmap-gpu-continuation.json`),
deterministic-profile report (local-only `../artifacts/roadmap-gpu-continuation-deterministic.json`),
and independent CPU verification (local-only `../artifacts/roadmap-gpu-continuation-validation.json`).
The latter rechecks immutable protocols, logs/timing hashes, actual arrays/state,
and decoded membership against the original PNGs. The last decimal digits of
the default error string differ slightly from subtracting its JSON-roundtripped
operands; both observations are retained. That rounding difference does not
change the failed original comparison.

## Workload and fault model

Both studies used the same fixed real-image development cohort: eight training
and four validation RGBA skins, 64×64 pixels. These were the first rows of the
previously exposed pilot archive, with recorded original image byte/RGBA hashes
and split membership. The tiny NPZ SHA256 is
`750621842f82b31fae270bc0e47be674bc42e329a0ea4c91291ca29e059ab86a`.
The source pilot archive SHA256 is
`c8d5f70fba65903de749b17cc77773607953e74154866cbedd898892abacd0bc`.
These data are not an unexposed release-test set.

The native dense models used latent dimension 128, RGBA-MSE reconstruction,
batch size two, seed 1976, two AE epochs, two discriminator epochs, two GAN
updates, and checkpoint interval one. An uninterrupted reference ran in a fresh
GPU process. For each phase, another fresh process raised a cooperative Python
exception immediately after the actual maintained checkpoint pointer had been
published at the first epoch/update of that phase. A separate process resumed
that run to the unchanged final budgets. The probe wrapped checkpoint
publication for fault injection; it did not substitute tiny model fixtures.

The injected exceptions were expected and handled by the probe, so those
interruption processes exited 0 after preserving a marker and published
checkpoint. The default study stopped after its resumed-AE comparison failed;
discriminator and GAN interruption comparisons were not attempted there. The
successful second study completed seven fresh processes: one reference plus
three interruption/resume pairs. Subsequent checkpoints retired earlier partial
snapshots; retained interruption markers and the launcher's recorded publication
states describe those earlier points.

## Compiler profile and interpretation

The second study set this exact environment value before each Python/JAX
process started:

```sh
XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0'
```

[OpenXLA's GPU determinism documentation](https://openxla.org/xla/determinism)
distinguishes live-autotuning kernel selection from execution-time
nondeterministic operations. It describes disabling autotuning and excluding
nondeterministic implementations as controls for those paths. Such controls can
change compilation choices, performance, and supported operations.

The two observations are consistent with compiler-policy contribution to the
default discrepancy. They do not establish that it was the exclusive cause or
that either individual flag alone is sufficient. There was no independent
one-flag ablation, repeated-seed/device study, or controlled performance study.
The earlier reference's first AE metric matched its resumed counterpart's first
metric, but a later strict comparison diverged; a seed alone did not establish
continuity in that tested default profile.

The recorded runtime was Python 3.14.8, Keras 3.15.1, JAX/JAXLIB 0.11.2, the
pinned CUDA 13 environment, and NVIDIA GB10 on Linux aarch64. Full distribution
versions and historical execution-source fingerprints are in the reports.
Historical `training.py` fingerprints identify those executed revisions; they
must not be replaced with the current source hash after additive policy changes.

## Current execution-policy recording

New maintained runs hash the exact `XLA_FLAGS` environment text and record
effective JAX x64 and default matmul precision. Resume rejects a changed recorded
policy before expensive data/model loading. The flags are startup configuration:
changing environment text after JAX/compiler initialization does not prove a
new compiler context. The digest checks text identity, including whitespace;
it does not introspect loaded kernels or normalize semantically equivalent flag
strings. Older run metadata without this field retains its narrower existing
compatibility checks and cannot retrospectively prove compiler-policy identity.

The latest policy-revision smoke report (local-only `../artifacts/roadmap-gpu-policy-smoke.json`)
records a new native 2/2/2 GPU run with all 84 checkpoint arrays byte-identical to
the historical deterministic reference and equal RNG/counters. A changed-flags
completed-run resume exited 2; same-flags reload exited 0. The launcher recorded
all 13 protected run files unchanged in both cases. Independent review rechecked
configuration/metrics/source hashes, the exact flags digest, retained log hashes,
and checkpoint byte equality. It does not reconstruct an unavailable pre-case
file inventory. This smoke validates the additive guard and accepted completed
reload; interrupted training is covered by the separate historical study.

Use the same flags at process startup for training and resumption, and retain
all data/run/configuration parameters. For example, with the locally prepared
tiny cohort and a new run path:

```sh
XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0' \
  scripts/gpu.sh train \
  images/results/roadmap-2026-10-02/gpu-continuation-deterministic/data/train_test.npz \
  images/results/continuation-demo \
  --device gpu --seed 1976 --batch-size 2 --encoded-dim 128 \
  --ae-epochs 2 --discriminator-epochs 2 --gan-steps 2 --checkpoint-interval 1

XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0' \
  scripts/gpu.sh train \
  images/results/roadmap-2026-10-02/gpu-continuation-deterministic/data/train_test.npz \
  images/results/continuation-demo \
  --device gpu --seed 1976 --batch-size 2 --encoded-dim 128 \
  --ae-epochs 2 --discriminator-epochs 2 --gan-steps 2 --checkpoint-interval 1 --resume
```

The second command resumes a compatible checkpoint or validates/reloads a
completed run. These commands do not themselves inject the study's cooperative
fault. Full study scripts, frozen protocols, logs, timings and retained run
artifacts live under `images/results/roadmap-2026-10-02/gpu-continuation/` and
`gpu-continuation-deterministic/`. Existing study paths must not be reused or
silently retried; a repeat requires new output paths and its own frozen protocol.

## Resource scope and remaining limits

Every recorded process exited 0 and remained below the 120-second per-process
and 840-second aggregate planning limits. The default study ended after 71.63
seconds; the complete deterministic study took 94.56 seconds. Observed maximum
external RSS was 3.193 GiB and 2.883 GiB respectively, below the 16 GiB host
budget. Completed-run allocator snapshots stayed below the 4 GiB device planning
budget. They are process-lifetime counters recorded by the runner after export
and before final validation, not an independent device peak after termination.
External RSS covers the full process. These are bounded observations, not
portable memory or throughput guarantees.

Continuation evidence is limited to one seed, one tiny real cohort, one tested
GPU/runtime, native dense topology, and cooperative post-publication exceptions.
It does not verify full-corpus or convolutional continuation, cross-device or
cross-version equality, interrupts during publication, SIGKILL, power loss, or
filesystem crash recovery. Uncheckpointed updates replay. No global GPU
continuity guarantee, creator-quality acceptance, or release-generalization
claim follows from this study.
