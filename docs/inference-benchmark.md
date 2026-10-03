# Isolated decoder inference measurements

On 2026-10-02, the seed-1976 dense and convolutional pilot bundles were measured
in fresh, sequential GPU processes on NVIDIA GB10 (driver 580.178.04, Linux
aarch64). Python 3.14.8, Keras 3.15.1, JAX/JAXLIB 0.11.2, and the pinned CUDA 13
requirements were used. Full package versions, fingerprints, all observations,
and memory snapshots are in
the benchmark evidence (local-only `../artifacts/roadmap-inference-benchmark.json`).

| Decoder | Parameters | Batch-16 median latency | Min–max | Process peak RSS | Allocator peak in use | Allocator peak pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense baseline | 17,530,384 | 3.917 ms | 2.090–5.731 ms | 1,439.8 MiB | 289.1 MiB | 518.0 MiB |
| Convolutional | 588,468 | 2.368 ms | 1.481–5.477 ms | 1,563.4 MiB | 386.3 MiB | 518.0 MiB |

Both models returned finite `(16, 64, 64, 4)` float32 output with identical output
hashes across their own repeated calls. This is a bounded local observation,
not evidence that convolutional models are generally faster or better. Their
encoder representations differ, so the two models did not receive semantically
equivalent latent vectors. The convolutional model had fewer parameters but
higher observed process and device-allocation peaks in this measurement.

## Protocol

Each process loaded one validated generation bundle and copied its first 16
stored encoder codes. Both bundles identify the same immutable pilot dataset
archive fingerprint:
`c8d5f70fba65903de749b17cc77773607953e74154866cbedd898892abacd0bc`.
Three untimed `decoder.predict(batch_size=16, verbose=0)` calls warmed compilation
and caches. Thirty subsequent calls were individually timed with
`time.perf_counter_ns`. Each timed section included prediction, `np.asarray`,
shape checking, and a finite-value reduction on completed host NumPy output.
SHA256 verification followed outside the timed section. No PNG encoding, latent
sampling, encoder inference, or disk output was timed.

The dense process finished successfully before starting the convolutional
process. Each command had a 120-second timeout; actual command wall times were
5.78 and 4.62 seconds respectively. Those whole-command times include startup
and warmup and are distinct from warm prediction latency. Settings were
`JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MPLBACKEND=Agg`,
`OMP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.

Linux process peak RSS comes from `/usr/bin/time` and covers the full process.
JAX device memory snapshots follow loading, warmup, and the 30 predictions.
Their peak counters include allocation during loading/JIT/warmup; they are not
per-call or training peaks. Pool reservations and allocated bytes measure
different things, and neither counter proves total system GPU consumption.
The driver reported total memory as unavailable, so this report does not infer
a physical device-memory capacity from allocator limits.

## Local reproduction

The benchmark script, raw JSON, logs, and external resource reports are local
experiment artifacts under
`images/results/roadmap-2026-10-02/isolated-inference/`.
They depend on the locally trained pilot bundles and are not a fresh-clone
training or inference guarantee. The checked evidence records the script,
lockfile, GPU requirements, bundle metadata, decoder, codes, actual latent bytes,
and output SHA256 values. To repeat after those artifacts exist, use a new
output filename and run each process sequentially:

```sh
/usr/bin/time -o new-baseline-time.txt \
  timeout --signal=TERM --kill-after=5s 120s \
  env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  scripts/uv.sh run --locked --with-requirements requirements-gpu.txt python \
  images/results/roadmap-2026-10-02/isolated-inference/benchmark.py \
  images/results/roadmap-2026-10-02/model-pilot-runs/seed-1976-baseline/bundle \
  new-baseline.json
```

After successful termination, repeat with `seed-1976-conv/bundle` and new
convolutional output paths. Existing output files are deliberately refused.

One process per model is insufficient to estimate variation between processes,
order effects, cold-start distribution, or other hardware behavior. These
measurements do not establish training-memory budgets, matched visual quality,
or an architecture adoption decision. Those require the controlled training
and creator-quality evidence described in the roadmap.
