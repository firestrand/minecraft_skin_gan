# Isolated spatial-model training resources

Six fresh GPU processes completed the matched AE-only pilot: dense and
convolutional models, each with seeds 1976, 2026 and 2027. Every process completed
exactly five AE epochs, zero discriminator epochs and zero GAN updates. The
controller finished in 190.63 seconds, below the frozen 720-second aggregate
cap; every command stayed below its 120-second cap and observed 16 GiB host /
4 GiB device-allocation planning limits. No run failed, retried, or changed the
frozen input/source/tool files.

On this NVIDIA GB10, convolutional workflows used fewer parameters and less
host memory, but higher device allocator peaks. The observed whole-command
wall times were also lower. These are three-seed observations on one hardware
profile, not portable speed/memory guarantees or an architecture adoption
result. The maintained dense baseline remains the adoption baseline pending
rendered creator review and the remaining quality gates.

| Architecture | Whole-command time, mean (range) | Whole-process peak RSS, mean (range) | Phase-end host highwater, mean (range) | Phase-end device in-use peak, mean | Phase-end device pool peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 36.79 s (35.86–37.45) | 3.210 GiB (3.170–3.233) | 2.883 GiB (2.878–2.888) | 0.726064 GiB | 1.002 GiB |
| Convolutional | 25.15 s (24.62–25.60) | 2.669 GiB (2.663–2.673) | 2.417 GiB (2.412–2.421) | 0.990372 GiB | 1.998 GiB |

The dense in-use peaks ranged from 0.726063728 to 0.726065159 GiB; convolutional
in-use peaks were 0.990371704 GiB for all three seeds. Pool peaks were identical
within each architecture. Full-precision per-run values and mean/min/max summaries
are in the verified resource evidence (local-only `../artifacts/roadmap-spatial-training-resources.json`).

| Architecture | Autoencoder parameters | Decoder parameters | Discriminator parameters |
| --- | ---: | ---: | ---: |
| Dense | 35,044,512 | 17,530,384 | 17,387,001 |
| Convolutional | 1,136,612 | 588,468 | 23,793 |

Both workflows instantiate their complete native model set, including the
untrained discriminator. Parameter totals alone therefore do not explain the
resource measurements. The reported timings include compilation, archive
loading, model setup, training, checkpointing, encoding and export.

## Frozen comparison

The protocol was written and fingerprinted before any GPU launch. It fixed the
real pilot archive `model-pilot-data/train_test.npz`, with 4,096 training and
1,024 previously exposed development-validation skins, each 64×64 RGBA uint8.
Its SHA256 is
`c8d5f70fba65903de749b17cc77773607953e74154866cbedd898892abacd0bc`.
The membership manifest, all current maintained Python sources, relevant legacy
model code, package/tool manifests and launcher/controller scripts were hashed.
The original pilot's split membership and input bytes were retained unchanged.

Other than architecture, seed and isolated output path, every configuration
field was identical: cold initialization, latent dimension 128, batch size 128,
RGBA-MSE reconstruction, five AE epochs, zero D/G budgets, bandwidth 3.16,
AE/D/G learning rates 0.001/0.001/0.00001, one discriminator update per GAN step,
and checkpoint interval 50. Each process ran the actual maintained training CLI.
There was no warm-start, fixture model, monkeypatch or earlier model/JIT cache
within that process. Order was dense-1976, conv-1976, dense-2026, conv-2026,
dense-2027, conv-2027; process order was not randomized.

The recorded hardware was NVIDIA GB10, driver 580.178.04, Linux aarch64. Runtime
versions were Python 3.14.8, Keras 3.15.1 and JAX/JAXLIB 0.11.2, using the locked
optional CUDA 13 dependencies. Every resolved package version and recorded
startup-policy/precision field matched across the six runs. Those records check
startup text and JAX precision settings, not the selected GPU kernel binaries. Settings were
`JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MPLBACKEND=Agg`,
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. `XLA_FLAGS` was explicitly empty:
this measured the default GPU compiler profile, not the separate deterministic
continuation profile.

## Per-run observations and historical numeric comparison

| Run | External wall time | Whole-process peak RSS | Phase-end device in-use peak | Full 1,024-image validation RGBA MSE | Change from earlier same-seed pilot MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense 1976 | 37.05 s | 3.228 GiB | 0.726065 GiB | 0.04197536 | −0.00037607 |
| Conv 1976 | 25.60 s | 2.670 GiB | 0.990372 GiB | 0.04084764 | −0.00005432 |
| Dense 2026 | 37.45 s | 3.170 GiB | 0.726064 GiB | 0.04207137 | +0.00047314 |
| Conv 2026 | 25.24 s | 2.663 GiB | 0.990372 GiB | 0.04165202 | −0.00019593 |
| Dense 2027 | 35.86 s | 3.233 GiB | 0.726064 GiB | 0.04185015 | −0.00024488 |
| Conv 2027 | 24.62 s | 2.673 GiB | 0.990372 GiB | 0.03954781 | +0.00016709 |

Mean validation RGBA MSE was 0.04196563 for dense (0.04185015–0.04207137) and
0.04068249 for convolutional (0.03954781–0.04165202). These numeric observations
do not score wearability or visible detail. The [earlier model pilot](experiment-results.md)
also measured visible RGB reconstruction and generated samples, and retained a
defer-adoption recommendation.

The earlier pilot used the same dataset, native architectures, seeds and phase
budgets, but reused a process across several models/losses. This fresh-process
study differs in compiler/cache/process context and includes the current
additive execution-policy recording revision. Numerical differences are retained
rather than interpreted as bitwise regression proof. The [GPU continuation
study](gpu-continuation.md) demonstrated that the default profile did not meet
its strict fresh-process continuity comparison; this resource study did not
isolate the cause of its own historical metric differences.

## Memory scope and independent verification

`/usr/bin/time -v` supplies whole-process peak RSS and wall time, including final
export and validation. A complete PPid-tree guard sampled process-family RSS at
one-second intervals, including uv worker-thread descendants; shared pages can
be counted multiple times. Its largest observed family RSS was 3.243 GiB.
Published checkpoint device highwaters were required, with both in-use and pool
keys checked against the declared cap. Missing counters would fail the study;
they were not treated as zero. Sampling is not a continuous physical-VRAM guard.

The final AE history snapshot precedes its final checkpoint publication,
encoder-code extraction, model/bundle export and final validation. Its host and
device peaks are lifetime highwaters for that fresh AE workflow, including
imports, archive load, full model initialization, compilation, earlier
checkpointing and all five AE epochs/validation passes. They are not the
incremental memory of one update. The runner's later allocator snapshot follows
export but precedes final validation; it matched the corresponding phase-end
in-use/pool peaks in all six runs. An independent device peak after termination
is not available. Allocator in-use and pool bytes are different quantities and
neither measures total system GPU consumption. Physical total GPU memory was
reported unavailable and was not inferred from allocator limits.

A separate CPU verifier rehashed every frozen input/source/tool file, checked
actual NPZ shapes/dtypes and manifest counts, compared exact configuration and
all runtime package/policy records, checked five ordered AE history entries and
completed counters, and confirmed all recorded validation metrics, exported
`(4096, 128)` float32 codes and saved checkpoint arrays are finite. Bundle and
checkpoint JSON/NPZ hashes, external resource reports, logs and terminal exit
records were verified. Per-batch losses/gradients are not retained by the runner,
so they are outside that finite-value evidence. CPU verification is excluded
from the GPU training-resource measurements.

## Local reproduction and boundaries

The frozen protocol, controller, independent verifier, observations, six run
directories, logs and resource reports are local artifacts under
`images/results/roadmap-2026-10-02/spatial-training-resources/`. They depend on the
existing prepared pilot corpus. Each command used `scripts/gpu.sh train` with
explicit fixed budgets/options; the evidence preserves the exact argument lists.
The recorded original controller invocation from the project root was:

```sh
python3 images/results/roadmap-2026-10-02/spatial-training-resources/execute.py
```

The existing outputs must not be reused: logs and run directories refuse
collisions, and a failure stops the controller without silent retries. To repeat,
copy the scripts to a new experiment directory, declare new run/log/output paths,
freeze a new input/source/tool protocol and execute the controller at its new
path. A repeat requires its own verified fingerprints and outcomes; these local artifacts do
not provide fresh-clone data access or a portable performance guarantee.

This closes the isolated AE-training resource comparison on the real pilot
cohort. It does not measure convolutional discriminator/GAN training,
full-corpus convolutional requirements, repeated-process uncertainty per seed,
other hardware/runtime profiles, matched latent semantics, human rendered
quality, or release generalization. Architecture adoption remains deferred;
combine these resource observations with existing inference/quality evidence
and the required creator rubric before escalation.
