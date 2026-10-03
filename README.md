# minecraft_skin_gan

A generative autoencoder (GAE) for Minecraft 1.8 RGBA skins. The original dense
encoder, decoder, discriminator, KDE sampling and image preprocessing are retained.

Requires Python **3.14.8**. The checked-in uv lock uses current Keras 3 and the
supported JAX CPU backend; a GPU is not required. Install `uv`/`uvx` using
[Astral's installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
The project wrapper pins uv **0.12.22**, including on machines with an older uv.

From a fresh clone:

```bash
scripts/uv.sh python install 3.14.8
scripts/uv.sh sync --locked --group dev
scripts/uv.sh build --no-sources --no-build-isolation
scripts/verify.sh
```

`verify.sh` checks formatting, lint, types, unit tests, line and branch coverage
(each must exceed 80%), then builds the wheel and source archive. Coverage reports
are `artifacts/coverage.json` and `artifacts/coverage.xml`; artifacts are generated,
not maintained source. To run just the test suite or validate a fresh non-editable
wheel install outside the checkout:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 scripts/uv.sh run --locked pytest
scripts/uv.sh run --locked python scripts/check_reproducibility.py
```

Linux x86_64 and ARM64 run the same gate in CI. Local verification uses ARM64.
The lock and declared build backend pin dependencies; `--locked` prevents implicit
lock changes. For deliberate dependency maintenance, run `scripts/uv.sh lock --upgrade`
and rerun both verification commands. Build artifacts do not include skin data or models.

## Existing workflow

Run commands from the project root. The download, filter and duplicate scripts
modify local image files; use your own backed-up dataset. Importing modules has no
network, training, deletion or generation side effects.

```bash
scripts/uv.sh run --locked python download_skins.py
scripts/uv.sh run --locked python sort_skins.py
scripts/uv.sh run --locked python remove_duplicates.py
scripts/uv.sh run --locked python create_skins_array.py
scripts/uv.sh run --locked python simple_gan.py
scripts/uv.sh run --locked python generate_skin.py
```

Defaults remain as in the legacy scripts: download IDs `[15148205, 15206629)` into
`images/skins/` with 20 workers; filter low head-color variation into `images/other/`;
deduplicate PNGs directly in **`images/`** (not recursively); create
`images/train_test.npz` from `images/skins/`, preserving `arr_0`/`arr_1`, the 80/20
split and random seed 1976. Downloader requests time out after 30 seconds; transport
and disk errors propagate, non-200 responses return `False`. The final partial batch
respects the stop ID. The preprocessing `main` functions accept optional source and
destination paths; downloader also accepts range, worker, batch and timeout options
as Python arguments. `sort_skins.should_filter_skin` exposes the original criterion.

`simple_gan.py` trains the original 64×64×4 / 128-dimensional GAE for 50 epochs,
then saves `models/autoencoder.keras`, `models/decoder.keras` and the comparison plot.
The comparison plot retains the original first-ten selection (indices 1–10), so
this entry point requires at least 11 test skins. The GAE class accepts other shapes
and retains all historical method names. The module functions
`approximateLogLiklihood` and `findNearest` retain their public names. `mean_log_likelihood` retains its historical fit-only / `None` behavior.
`generate_skin.py` draws 100 uniform 128-dimensional samples and writes clipped,
rounded 64×64 RGBA PNGs named `images/results/gae_0.png` through `gae_99.png`.
`generate_skin.generate_skins(model_path, output_directory, count=100)` returns their
paths; `load_decoder(model_path)` loads the decoder independently.

## Model migration

Keras 3 replaces the abandoned PlaidML backend. The default is `KERAS_BACKEND=jax`;
an explicitly set backend is respected and needs its own compatible installation.
For headless runs set `MPLBACKEND=Agg`. No other environment variables are required.
Models use `.keras`; epoch checkpoints use `models/weights_{name}.{epoch:02d}.weights.h5`.
Training resumes the highest numbered checkpoint, including historical
`weights_mnist_{name}.*.hdf5` and `weights_{name}.*.hdf5`, and trains remaining epochs. Checkpoint resume restores weights; optimizer state
is not guaranteed to resume in a fresh process.
The discriminator has a separate optimizer, is trainable for its own updates, and
is frozen during generator updates, as required by current Keras optimizer semantics.

`load_decoder` accepts a trusted legacy HDF5 model named `.mdl`, loading a temporary
`.h5` copy because Keras dispatches by extension. The default loader falls back to
`models/decoder.mdl` when `decoder.keras` is absent. Save the loaded model to `.keras`
to complete migration. Legacy SavedModel directories or PlaidML-specific artifacts
must first be exported as full HDF5 models using their original compatible backend;
they cannot be directly loaded by Keras 3. Only load trusted model artifacts.

No legacy corpus, trained weights or executable test baseline was supplied for the original modernization. A later local archive recovery supplied images and a real GPU training run; see the roadmap workflow below. Tests
verify numerical contracts, PNG/NPZ/model roundtrips, real CPU weight updates and
failure paths; they cannot establish equivalent skin quality after a full retrain.
The JAX backend avoids PyTorch’s incompatible SymPy/mpmath dependency constraints.
All runtime, transitive and development dependencies use current stable releases
as verified on 2026-10-02. The [roadmap](docs/roadmap.md) records the baseline, scope and remaining quality gates.

## Reproducible local workflow

The additive `skin-gan` command preserves the root scripts' historical behavior while
providing versioned artifacts and explicit paths. Install with `scripts/uv.sh sync --locked --group dev`;
run `scripts/uv.sh run --locked skin-gan --help` for available commands. A wheel install also
provides `skin-gan` outside the checkout; `python -m minecraft_skin_gan` is equivalent.

```bash
scripts/uv.sh run --locked skin-gan prepare /path/to/skins /path/to/new-dataset --provenance 'Source description'
scripts/uv.sh run --locked skin-gan train /path/to/new-dataset/train_test.npz /path/to/new-run --device cpu --ae-epochs 10 --discriminator-epochs 10 --gan-steps 830
scripts/uv.sh run --locked skin-gan generate /path/to/new-run/bundle /path/to/new-samples --count 16 --seed 1976
scripts/uv.sh run --locked skin-gan evaluate /path/to/new-run/bundle /path/to/new-dataset/train_test.npz /path/to/new-evaluation --autoencoder /path/to/new-run/models/autoencoder.keras
```

New output directories must not already exist. Preparation records invalid files,
decoded-content groups, stable split membership, provenance, and separate semantic
and archive byte hashes. Exact RGBA duplicates stay in the same split. Perceptual
matches are review candidates and do not change split membership automatically.
The new split policy is explicit and differs from the historical seed-1976 split.

Generation bundles contain the decoder and its fitted latent-sampling data, avoiding
the historical standalone uniform-sampling mismatch. `--sampler empirical` samples
stored encoder codes; `--sampler legacy_uniform` explicitly uses the old latent
distribution. `--bandwidth` changes KDE noise. Seeds reproduce output within the same
pinned runtime/device and compiler policy; cross-device bit identity is not promised.

Training uses bounded image batches, validation-selected AE weights, separate phase
budgets and learning rates, isolated checkpoints, and host/device memory observations.
New configurations record resolved package versions, OS/machine and device kind;
metrics record model parameter counts. Resume rejects a changed package-version
snapshot when one was recorded. Older runs retain their narrower version checks.
New runs also fingerprint `XLA_FLAGS` without storing its contents and record
effective JAX x64/matmul precision. Resume rejects a changed recorded execution
policy before decoding the archive or constructing models. Set compiler flags
before starting Python; a seed and package versions alone do not fix GPU
autotuning choices.
GAN checkpoints default to every 50 updates (`--checkpoint-interval`), retaining the
latest snapshot. `--resume` requires matching configuration, data and recorded runtime;
updates after the published checkpoint replay. `--warm-start /path/to/prior-run` copies
compatible model weights and resets optimizers, counters and sampling RNG. The maintained
runner's shuffled AE batches, fresh discriminator fakes, and combined real/fake updates
are an explicit new training policy; the legacy `GAE.train()` policy remains available.

Evaluation reports development metrics, exact cross-split overlap, nearest-training
examples and fixed-seed sheets. It records unverified human quality and format rules
explicitly. A supplied adjacent manifest must match the actual archive byte hash.
Final generalization claims require genuinely unexposed data and an agreed rubric.

For additional source images, the [bounded acquisition command](docs/acquisition.md)
validates responses and writes completed/failed/skipped outcomes to a manifest:

```bash
scripts/uv.sh run --locked skin-gan acquire /path/to/new-downloads --url 'https://provider.example/skins/{skin_id}.png' --start 0 --stop 100 --provenance 'Permitted source description'
```

The URL is a placeholder: supply a permitted provider endpoint. The ID range is
half-open; partial failed outcomes return exit 1. Existing output directories are
refused. Tests mock the network; no live provider was verified.

Preview and gallery commands accept `--alpha-mode original` (default) or
`--alpha-mode opaque-base`. The latter affects embedded base textures only;
source, downloaded and ZIP-exported PNG pixels remain unchanged. Record the same
profile across quality comparisons; see [renderer details](docs/skin-format.md).

### Controlled experiments

Save this as `/path/to/work/plan.json`, with an existing prepared archive at
`/path/to/work/dataset/train_test.npz`. Paths inside the plan resolve relative to the
plan file, including when the installed command runs outside the checkout.

```json
{
  "schema": "minecraft-skin-gan.experiment-plan/v1",
  "data_path": "dataset/train_test.npz",
  "output_path": "reports/ablation",
  "configs": [
    {
      "name": "ae-only",
      "training": {
        "data_path": "dataset/train_test.npz",
        "run_path": "runs/ae-only",
        "ae_epochs": 1,
        "discriminator_epochs": 1,
        "gan_steps": 0,
        "device": "cpu",
        "seed": 1976
      }
    },
    {
      "name": "ae-gan",
      "training": {
        "data_path": "dataset/train_test.npz",
        "run_path": "runs/ae-gan",
        "ae_epochs": 1,
        "discriminator_epochs": 1,
        "gan_steps": 10,
        "device": "cpu",
        "seed": 1976
      }
    },
    {"name": "narrow-kde", "source_run": "ae-gan", "bandwidth": 1.0},
    {"name": "empirical", "source_run": "ae-gan", "sampler": "empirical"}
  ]
}
```

```bash
scripts/uv.sh run --locked skin-gan experiment /path/to/work/plan.json
scripts/uv.sh run --locked skin-gan experiment /path/to/work/plan.json --resume
```

Both training variants use identical pretraining budgets; only GAN duration changes.
The two sampling variants reuse the `ae-gan` bundle without retraining. For sampling
an existing model, replace a candidate's source with `"bundle_path": "runs/prior/bundle"`.
Each candidate requires exactly one of `training`, `bundle_path`, or `source_run`;
`source_run` must name a training candidate in the same plan. Training entries must
declare both paths and all three phase budgets; remaining fields accept the maintained
`TrainingConfig` options, including architecture, loss and discriminator update ratio.

The full plan, dataset manifest and output collisions are checked before training.
Training directories must be distinct and separate from the report directory. All
candidates share `sample_count` (default 16) and `sample_seed` (default 1976), with at
most 64 candidates and 64 samples per candidate. Runs execute sequentially; resource
budgets do not promise a wall-clock limit. `--resume` requires the same plan, dataset
and pinned runtime, preserving the training runner's checkpoint guarantees.

`reports/ablation/experiments.json` records the exact held-out cohort, dataset hash,
resolved configurations, versions, training metrics and run paths, reconstruction
errors, generated-image/alpha statistics and nearest examples from all training images.
Each candidate also saves latent probes and generated arrays for comparison. These
are development measurements; choosing a model still requires the visual rubric.

Read-only curation is the default:

```bash
scripts/uv.sh run --locked skin-gan curate /path/to/skins
scripts/uv.sh run --locked skin-gan curate /path/to/skins --apply /path/outside/source/quarantine
scripts/uv.sh run --locked skin-gan undo /path/outside/source/quarantine/manifest.json
```

Quarantine checks source/keeper hashes and destination collisions; undo restores original
bytes. Ordinary failures roll back moved files. This is not a crash-proof filesystem
transaction; keep backups. `--policy head-filter` explicitly selects the historical
head-variance policy. These commands never curate the recovered corpus automatically.

For supported NVIDIA CUDA 13 systems, `scripts/gpu.sh` runs the same commands in the
optional pinned environment from `requirements-gpu.txt`, leaving the CPU lock unchanged:

```bash
scripts/gpu.sh train /path/to/data/train_test.npz /path/to/new-gpu-run --device gpu --ae-epochs 10 --discriminator-epochs 10 --gan-steps 830
```

This profile forces CUDA and disables JAX memory preallocation. Hardware/driver
requirements follow the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html).
The CUDA 13 wheels require Linux x86_64/aarch64, an NVIDIA SM 7.5+ GPU and Linux
driver 580+. The tested profile is Linux aarch64 / NVIDIA GB10 / driver 580.178.04.
The full 132,810-image dense run at batch 128 used about 6.49 GiB host peak RSS
and 0.90 GiB peak device allocation. For these experiments use a **16 GiB host /
4 GiB device planning budget**, monitoring the reported high-water values. The
nine-run AE pilot stayed below 4.79/2.55 GiB respectively. The sequential nine-run
GAN pilot declared a separate 16/8 GiB budget and reached 8.06/6.90 GiB, including
cached compilations across models. Allocator statistics omit
some driver/system memory, and sequential runs retain JIT caches; these figures
are not minimum hardware guarantees. Other GPUs need their own smoke/memory check.
Unit tests do not require CUDA. Generated datasets, bundles and experiment outputs
remain local; code licensing does not establish permission to redistribute recovered skins.

The default native GPU profile failed a strict interrupted/resumed comparison.
An explicit compiler profile matched all tested checkpoint arrays exactly across
three fresh-process resumes on GB10:

```bash
XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0' scripts/gpu.sh train /path/to/data/train_test.npz /path/to/new-run --device gpu
XLA_FLAGS='--xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0' scripts/gpu.sh train /path/to/data/train_test.npz /path/to/new-run --device gpu --resume
```

Use the identical flags for initial training and resume. This tested dense/tiny-cohort
profile is narrower than a general GPU continuation guarantee; see
[continuation evidence](docs/gpu-continuation.md). [Isolated full-corpus phase measurements](docs/training-memory.md)
document AE, discriminator and GAN resource use separately.

Remaining experiments, features and acceptance criteria are described in
[the roadmap](docs/roadmap.md). Execution notes and generated evidence are local-only.
Measured study results are in [experiment-results.md](docs/experiment-results.md).
Evaluation now emits [renderer alpha/seam diagnostics](docs/skin-format.md);
use `skin-gan evaluate ... --model-type slim` for slim arms. Diagnostics do not
certify game acceptance. Human comparisons use the draft [quality rubric](docs/quality-rubric.md).

## Skin scoring viewer

Open the [prepared local viewer](images/results/roadmap-2026-10-02/scoring-viewer/index.html) to inspect 32 generated candidates and collect ratings. One HTML file contains all candidates. It runs offline in Firefox, with mouse rotation, a white model background, zoom, visible ratings, browser drafts and CSV/JSON exports. Use **Save rated HTML** to keep scores inside a portable copy of the viewer.

```bash
scripts/uv.sh run --locked skin-gan review path/to/skins path/to/new-review --seed 1976
```

See the [viewer guide](docs/scoring-viewer.md) for anonymous review, backups and rendering profiles. The prepared packet depends on local experiment outputs; fresh clones can create packets from their own 64×64 PNGs.
