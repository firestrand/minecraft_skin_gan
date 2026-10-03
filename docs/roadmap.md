# Minecraft Skin GAN: improvement and feature roadmap

Reviewed: 2026-10-02. Scope: the modernized code, recovered image corpus, historical baseline, and additive roadmap implementation. This document sets direction and acceptance criteria; it does not certify implementation completion. Consult the acceptance register (local-only `roadmap-progress.md`) and underlying evidence for item status. Existing `LOCAL-` IDs remain stable.

Planning reference: Development Plan Creation Guide v2.7 (2026-09-27); technical constraints: Python Code Standards (reviewed 2026-09-29). This is a strategic roadmap with deferred research decisions, not a dispatch-ready implementation plan. Expand each remaining work packet against those standards before executing it.

## Recommended direction

Finish verifying the implemented generation, dataset, training and creator interfaces, then improve image quality through controlled experiments. Reproducible bundles and stable manifests now exist; the remaining priority is evidence about output usefulness. A new framework or a large web application would add little value before that evidence exists.

The immediate product goal should be **generate a reproducible batch of usable 64×64 Minecraft skins from a saved local model**, with a preview and traceable training provenance. “Usable” needs an agreed visual and skin-format acceptance rubric; a valid PNG and a low reconstruction error do not establish it.

## Historical baseline and current implementation

The table below preserves the original review baseline. Its 55-test figures and first training run are historical, not the current package's full verification results. The latest integrated and fresh non-editable-wheel gates both recorded **323 passing tests, 96.73% line coverage and 92.36% branch coverage**, with lint, types, build and locked installation passing (integrated log (local-only `../artifacts/viewer-single-file-verify.log`), fresh-wheel log (local-only `../artifacts/viewer-single-file-reproducibility.log`)). These gates verify implemented behavior; they do not establish generated-skin quality or completion of every roadmap item.

| Area | Observed state | Evidence |
| --- | --- | --- |
| Toolchain | Python 3.14.8, pinned uv, Keras 3/JAX, Ruff, ty, pytest, and a locked dependency set. The existing modernization audit reports current stable dependency versions at its audit time. | [Manifest](../pyproject.toml), [lockfile](../uv.lock), dependency audit (local-only `../artifacts/dependency-audit.json`) |
| Build and checks | Earlier modernization verification recorded a clean build and reproducible artifacts. This review reran the suite: **55 tests passed**, with 366 upstream Keras/NumPy deprecation warnings. | Verification log (local-only `../artifacts/verify.log`), reproducibility log (local-only `../artifacts/reproducibility.log`), command below |
| Unit coverage | Earlier measured coverage: **98.68% lines, 94.59% branches** across the configured source modules. Coverage was not recollected during this document-only review. | Final snapshot (local-only `../artifacts/final-snapshot.json`), [coverage gate](../scripts/check_coverage.py) |
| Recovered data | 132,840 decodable 64×64 skin PNGs; another 27,950 images in the archive's `other` directory. Decodability was checked, but uniqueness, skin-layout correctness, and redistribution rights remain unverified. | Local archive inventory described below |
| Real training | 106,272 training images and 26,568 held-out images; 10 autoencoder epochs, 10 discriminator epochs, and 830 GAN updates on `cuda:0`, taking about 327 seconds. Sixteen samples and saved models were produced. | Local [training metrics](../images/results/training-2026-10-02/metrics.json) |
| Output quality | Sample sheets show recognizable atlas structure and varied colors, but blurry details. Reconstruction sheets also lose detail. This is a visual finding, not a measured causal explanation. | Local [generated sheet](../images/results/training-2026-10-02/generated_contact_sheet.png), [originals](../images/results/training-2026-10-02/held_out_originals.png), [reconstructions](../images/results/training-2026-10-02/held_out_reconstructions.png) |

The training artifacts above are local ignored experiment outputs, not assets available from a fresh clone. The recovered corpus and `image-inventory.json` reside under `~/Downloads/minecraft_skin_gan_extracted_nrq9341_/`. Preserve them as the source baseline; do not silently clean or resplit that corpus.

### Current state and next priorities

| Workstream | Current evidence | Next decision or verification |
| --- | --- | --- |
| Generation and data (LOCAL-01–03) | Versioned bundles, deterministic grouped manifests, and reversible curation have implementation and fixture evidence; repeated corpus preparation and seeded generation were checked locally. | Preserve compatibility tests and record evidence against the exact integrated revision. |
| Training and runtime (LOCAL-04/06/07/12) | Maintained commands, CPU checkpoint-v2 parity, pinned GPU profile and fresh-wheel gates pass. Full-corpus fresh isolated phase workflows completed 10 AE / 10 discriminator epochs / 830 GAN updates within declared budgets. Default GPU continuation failed strict parity; an explicit deterministic profile passed all three phase interruptions exactly on a small real cohort. | Preserve the qualified [memory](training-memory.md) and [continuation](gpu-continuation.md) evidence and execution-policy checks. Other architectures, devices and workloads need separate measurements; no general GPU equality guarantee is established. |
| Evaluation and experiments (LOCAL-05/08–11/15) | Sampling comparisons, three-seed architecture/loss and duration/ratio pilots, isolated inference and palette feasibility studies are recorded. Radius-2 review found 61,340 cross-split similarity candidates. Sourced renderer diagnostics pass; the creator rubric remains a draft. | Agree and collect blinded human judgments, review similarity policy and verify actual game acceptance. Negative results support retaining the baseline; no architecture-quality improvement is established. |
| Creator workflow (LOCAL-13/14/16) | Preview UV fixtures and Firefox controls, byte-preserving gallery exports and region-compositing tests pass integrated packaging gates. Optional opaque-base previews preserve exported pixels. | Fix the rendering profile for comparisons; assess output coherence and user usefulness separately from pixel preservation. |
| Conditional generation (LOCAL-17) and release | The [annotation audit](annotation-availability.md) found metadata candidates in recovered PNGs; semantic labels and source permissions remain unverified. The reserved 27,950-image `other` cohort has no exact RGBA overlap with current training/validation, but radius-2 similarity scanning linked 3,303 `other` files to current data. Historical exposure and representative eligibility remain unresolved. | Review candidate annotation categories and provenance privately before choosing a vocabulary or seeking new labeled data. Follow the [release-candidate eligibility protocol](release-data-candidates.md) before evaluation; keep LOCAL-17 pending verified labels/provenance. Do not use the reserved cohort for candidate tuning. |

The maintained GPU run took 524.73 seconds inside the process, with process-lifetime peak host RSS of about 6.49 GiB and device peak allocation of about 0.90 GiB ([configuration](../images/results/roadmap-2026-10-02/training-gpu-v2/configuration.json), [metrics](../images/results/roadmap-2026-10-02/training-gpu-v2/metrics.json)). These are measurements on one machine, not portable minimum requirements or isolated phase peaks. Its final validation RGBA MSE was 0.0170055; this is reconstruction evidence, not a wearability score. Do not compare it directly with the historical run's differently prepared cohort.

Implementation follow-up: the original inventory checked decodability of files named `.png`. The new strict PNG-content validator accepted 132,810 images and identified 30 decodable files containing another image format; it recorded exclusions and preserved all source files. See the implementation acceptance register (local-only `roadmap-progress.md`) for current evidence and unfinished work.

The untrained reconstruction MSE was measured on the first 4,096 held-out images (0.20617), while the reported trained MSE used all 26,568 (0.01614). These are different cohorts, so they do not support a precise percentage improvement claim. The held-out set has now been inspected and can serve as development validation with that exposure recorded. Reserve genuinely unexposed data before final release evaluation; resplitting already trained-on or inspected images cannot retroactively make them untouched. The absence of a final test set need not block development experiments, but it limits claims about generalization.

## Historical findings that shaped the priorities

These findings apply to the retained root scripts and original run. The additive `minecraft_skin_gan` package addresses several of them through explicit bundles, manifests, isolated outputs, commands and checkpoints. Use these as regression requirements rather than instructions to implement duplicate solutions. See [current commands](../README.md#reproducible-local-workflow) and the acceptance register for remaining verification.

| Finding | Consequence | Priority |
| --- | --- | --- |
| `GAE.generate()` samples a KDE fitted to encoder codes; `generate_skins()` instead samples uniform 128-dimensional vectors. The decoder alone does not carry the trained sampler. | Standalone generation does not reproduce the distribution used in training. | P0 |
| Dataset preparation uses filesystem enumeration order before the seeded split and stores no filename/hash manifest. | A fixed seed alone cannot guarantee identical membership across machines. Duplicate leakage is unknown. | P0 |
| Preparation stacks the corpus in memory; training converts whole splits to float32 and combines real/generated discriminator examples with `vstack`. Peak memory has not been measured. | The demonstrated GPU run does not establish a practical memory requirement for other machines. | P1 |
| Duplicate cleanup deletes files on a perceptual-hash match; sorting moves files. Cleanup defaults also differ from preparation/download defaults. | Cleanup is difficult to audit or undo, and perceptually similar skins may have different meaningful pixels or alpha. | P0 |
| The successful GPU run uses an ignored script with callback monkeypatching and a working-directory change. Production training has hardcoded paths and budgets. | Repeating the experiment requires undocumented orchestration rather than a maintained interface. | P0 |
| `train(epochs=...)` controls the two pretraining phases, while the GAN budget is always `floor(image_count / batch_size)`. Early stopping monitors training loss, with no validation data. | The training budget is surprising, and stopping is not based on generalization. | P1 |
| Checkpoint selection uses the highest numbered filename. GAN progress, dataset identity, and RNG continuity are not recorded by the core runner. | True continuation after interruption is not established. | P1 |
| Generation writes fixed filenames and assumes a fixed latent width/output shape. No installed console entry points are declared. | Batch generation can overwrite results and is awkward outside the checkout. | P1 |
| Current quantitative results focus on reconstruction MSE and discriminator training metrics. | Neither directly demonstrates crisp, diverse, wearable generated skins. | P1 |

Evidence: [training/model implementation](../simple_gan.py), [generation](../generate_skin.py), [dataset preparation](../create_skins_array.py), [duplicate cleanup](../remove_duplicates.py), [filtering](../sort_skins.py), and [experiment runner](../images/results/training-2026-10-02/run_training.py).

## Prioritized roadmap

IDs prefixed `LOCAL-` are proposed tracking IDs, not existing product requirements. Effort is relative: S = focused change, M = coordinated pipeline work, L = model or product investigation. These are sequencing aids, not delivery dates.

### Phase 1 — Trustworthy generation and data

| ID | Improvement | Acceptance criteria | Dependencies / effort |
| --- | --- | --- | --- |
| LOCAL-01 | **Versioned generation bundle:** decoder, latent codes or an explicit sampler representation, KDE configuration, shape/normalization metadata, and model/data fingerprints. Add seed and count controls. | A fresh process loads the bundle and reproduces a fixed latent probe within documented tolerances; seeded samples agree within the same pinned environment. Reject incompatible shapes or missing sampler state before writing files. Preserve explicit legacy uniform sampling. | First / M |
| LOCAL-02 | **Versioned dataset manifest:** stable ordering, original path, content hash, decoded shape/mode, split membership, validation reason, and source provenance. Split exact-duplicate groups together; report perceptual near-duplicate candidates across splits. | Repeated preparation from the same inputs yields identical manifests and membership; no exact-content group crosses splits. Record the near-duplicate detection configuration and review grouping policy before changing membership; perceptual matches are not automatic deletions or labels. Invalid input is reported explicitly. Keep historical `arr_0`/`arr_1` archives and legacy split behavior available. | Independent of LOCAL-01 / M |
| LOCAL-03 | **Reversible curation:** one explicit dataset root; dry-run reports; exact RGBA-content duplicates separated from perceptual-similarity candidates; quarantine and undo manifests. Keep the historical head-variance filter as a named policy. | Default inspection changes no images. Every proposed move has a reason and keeper where applicable. Applying and undoing a fixture operation restores identical bytes; destination collisions are handled explicitly. | LOCAL-02 / M |

The first vertical slice was LOCAL-01: load the saved decoder and recovered latent KDE, generate a seeded batch in a fresh process, and record the bundle used. Preserve its sampler parity and output-collision tests while extending the pipeline.

### Phase 2 — Reproducible experiments and honest evaluation

| ID | Improvement | Acceptance criteria | Dependencies / effort |
| --- | --- | --- | --- |
| LOCAL-04 | **Maintained training runner:** explicit data/run paths, seed, batch size, separate autoencoder/discriminator/GAN budgets, learning rates, CPU/GPU policy, and isolated outputs. Record resolved config, versions, device, data fingerprints, phase timings, and metrics. Measure peak host/device memory for preparation and each training phase on the real corpus. | A small CPU smoke run and a documented GPU run complete without changing cwd or monkeypatching callbacks. Two runs do not share checkpoints or overwrite artifacts. Missing data/device errors occur before expensive work. Publish measured memory requirements and a supported memory budget; optimize allocation only if measurements justify it. | LOCAL-01, LOCAL-02 / M |
| LOCAL-12 | **Installed prepare/train/generate commands:** explicit paths, seed, count, model bundle, output directory, and collision policy. | Commands work outside the checkout; help describes actual defaults; inputs fail clearly; generation does not silently overwrite existing skins. Keep the existing callable APIs and script defaults compatible. | LOCAL-01, LOCAL-04 / M |
| LOCAL-05 | **Evaluation protocol with separate development and release gates:** use a documented development validation set, validation-driven autoencoder selection, fixed sample latents, and paired before/after measurements on the same cohort. Report AE-only and AE+GAN results separately. | **Development metrics milestone:** versioned reports/sheets record exposure, visible RGB reconstruction, alpha statistics, diversity, nearest-training examples and exact overlap. It enables explicitly provisional ablations. **Full development gate:** additionally document and review a nonzero near-duplicate threshold/candidate policy, implement sourced skin-format/alpha/seam checks, and record a human quality rubric. A distance-zero scan is not evidence that meaningful near-duplicates are absent. **Release gate:** evaluate a selected candidate on genuinely unexposed, disjoint final-test data before making final generalization claims. | LOCAL-02, LOCAL-04 for development; unexposed data for release / M |
| LOCAL-06 | **Resume with defined semantics:** checkpoint all phases, progress, config/data identity, optimizer and RNG state where supported. Separate warm-start from exact continuation. | Interrupt each phase and resume in a fresh process. Compare against an uninterrupted small run with explicit tolerances; reject incompatible configurations. If exact continuity cannot be supported, describe the narrower guarantee honestly. | LOCAL-04 / M |
| LOCAL-07 | **Supported GPU environment:** document a pinned optional CUDA training environment, hardware/driver requirements, and memory settings while retaining portable CPU installation. | Fresh CPU installation passes normal checks; supported GPU installation executes a tiny training and reload smoke test. Record both environments and avoid making GPU availability a unit-test requirement. | LOCAL-04 / S–M |

Keras supports configurable [early stopping](https://keras.io/api/callbacks/early_stopping/) and [checkpointing](https://keras.io/api/callbacks/model_checkpoint/). Do not assume a weights checkpoint can never contain optimizer state; verify restoration for this project's freshly constructed models and all training phases. The optional GPU profile should follow the [JAX installation requirements](https://docs.jax.dev/en/latest/installation.html).

If the measured memory use exceeds the supported budget, compare bounded image loading, chunked preparation, and batched generation/pretraining against the existing path. Require before/after memory and runtime measurements plus decoded-byte, normalization, and split-membership equivalence checks. Do not make a streaming rewrite a prerequisite without evidence of a bottleneck.

Suggested evaluation rubric:

- File correctness: dimensions, RGBA channels, finite values before export, and round-trip PNG integrity.
- Minecraft correctness: base-layer and overlay alpha rules, body-part placement, and seam checks against verified layout fixtures. The precise format rules need a documented source before enforcement.
- Visual quality: blinded comparisons of crispness, coherent body parts, palette consistency, and wearability using flat sheets and a character preview.
- Variety and copying: exact output duplicates, within-batch similarity, and nearest training examples. Similarity metrics flag review candidates; they do not prove originality.
- Operational quality: generation latency, peak memory, bundle size, and reproducibility within the recorded environment.

Set numeric quality thresholds after measuring the baseline and agreeing the rubric. Do not replace judgment with discriminator accuracy, pixel-space KDE likelihood, or a generic perceptual score alone.

### Phase 3 — Improve image quality through bounded experiments

Run provisional experiments after LOCAL-05's development metrics milestone, using the same dataset version, seed set, evaluation protocol, and recorded resource budget. Explicitly record missing format/human checks; full development acceptance and release claims require their respective remaining gates. Change one major factor at a time. Retain the dense model as the regression baseline and preserve existing architecture assertions for its legacy mode.

Before each study, freeze the source archive hash and any subset membership/hash, the shared validation cohort, training seed set (at least three seeds for architecture adoption), sample seeds/latents, phase budgets, batch size and hardware profile. Start with a bounded pilot, recording explicit per-run epoch/update limits, aggregate compute/time and memory caps, and a stop condition before launch. Record parameter count, wall time, memory, failures and metric variation across seeds alongside sheets. Sampling-only bandwidth/empirical comparisons reuse the same model; compare AE-only and AE+GAN using the same pre-GAN checkpoint and latent probes. A one-factor study keeps other settings fixed; any follow-up combination gets its own declared comparison. Escalate to a full-corpus study only when the pilot meets the predeclared practical-benefit criterion within budget.

Common latent probes are meaningful for decoders sharing an encoder space. Across independently trained dense/convolutional encoders, use fixed sampling seeds, equal sample counts and the same held-out images; do not claim the latent coordinates represent identical concepts.

Study completion means a reproducible report with results, limitations and an adopt/reject/defer recommendation. A negative result completes the investigation; it does not satisfy adoption criteria. Choose numerical improvement and acceptable resource thresholds from baseline measurements before evaluating candidates, and record the owner's visual-quality judgment separately. Keep the baseline if evidence is inconclusive.

| ID | Experiment | Decision evidence | Dependencies / effort |
| --- | --- | --- | --- |
| LOCAL-08 | **Training and sampling ablations:** AE-only versus AE+GAN; explicit GAN duration; discriminator/generator update ratio; KDE bandwidth sweep and empirical encoder-code sampling. | Paired validation results and fixed-seed sheets establish whether adversarial refinement or sampling improves detail/diversity. Include nearest-training checks to catch memorization as bandwidth decreases. | LOCAL-05 / M |
| LOCAL-09 | **Transparency-aware objectives:** compare existing RGBA MSE with visible-color and alpha-aware losses or masks. | **Study:** report paired metrics and rendered comparisons, including instability or negative results. **Adoption:** demonstrated visual/alpha benefit without unexplained detail loss or unstable training under the declared budget. Historical loss remains selectable. | LOCAL-05 metrics milestone, baseline LOCAL-08 / M |
| LOCAL-10 | **Spatial model variant:** a compact convolutional encoder/decoder and discriminator, with an explicit architecture identifier. Investigate atlas/body-part boundaries rather than assuming every neighboring pixel belongs to neighboring surface geometry. | **Study:** compare multiple seeds with the dense baseline; report quality, memory, latency and parameter count with a recommendation. **Adoption:** demonstrated benefit under the agreed budget and rubric; otherwise retain the dense baseline. Existing bundles still load. | LOCAL-05 metrics milestone, LOCAL-08 / L |
| LOCAL-11 | **Discrete pixel-art representation:** investigate quantized latents or palette-aware representations if controlled experiments still produce blur. | **Study:** bounded feasibility report measures crispness, copying risk and palette diversity, allowing rejection. **Adoption:** evidence of crisper outputs without unacceptable copying or palette collapse under the declared thresholds. No production migration on inconclusive evidence. | LOCAL-09, LOCAL-10 results / L |

The [KernelDensity API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) supports sampling and bandwidth control, so LOCAL-08 can begin with the current stack. [VQ-VAE](https://arxiv.org/abs/1711.00937) is a research reference for LOCAL-11, not evidence that it will solve this dataset's quality problems. Diffusion or a framework rewrite should remain a later research option rather than the default next step.

### Phase 4 — Features for skin creators

| ID | Feature and user benefit | Acceptance criteria / prerequisites | Effort |
| --- | --- | --- | --- |
| LOCAL-13 | **Local character preview:** front/back/rotating views with visible base and overlay layers. | Mapping matches verified atlas fixtures, including supported arm variants. Document model-type selection when it cannot be inferred. Accept arbitrary local PNGs as well as generated outputs. | M |
| LOCAL-14 | **Batch gallery and export:** seeds, favorites, side-by-side candidates, and PNG/ZIP export with provenance metadata. | Selecting/exporting a favorite reproduces its pixels; exports retain valid dimensions and alpha. Build on LOCAL-12/13 without requiring an account or hosted service. | M |
| LOCAL-15 | **Diversity and palette controls:** a visible sampling control and optional palette constraints. | Controls have measured effects and clear limits; reduced variation is not advertised as better quality. Depends on sampling experiments and evaluation. | M |
| LOCAL-16 | **Preserve selected body parts:** lock an existing head, clothing region, or overlay while generating alternatives. | **Compositing prototype:** verified classic/slim masks preserve selected base/overlay RGBA pixels exactly in the exported image; disclose that unlocked pixels come from the candidate and seams/coherence are not guaranteed. **Later conditional adoption:** paired rendered evaluation demonstrates coherent unlocked regions and acceptable seams. Conditional generation is a separate experiment. | L |
| LOCAL-17 | **Theme or text-guided generation:** use creator-selected concepts rather than browsing random candidates. | First review the existing [metadata candidates](annotation-availability.md) privately for useful annotation schemas, permitted uses and agreement with visible skins; field presence and filenames do not establish labels or rights. Requires provenance-reviewed, meaningfully labeled data and held-out conditional evaluation. Vocabulary, label creation and model choice remain unresolved. | L / exploratory |

LOCAL-13 can proceed beside model experiments once format fixtures are established. LOCAL-14 should wait for reproducible generation. LOCAL-16/17 require separate design and data decisions; they are ideas, not promises about the current model.

For LOCAL-17, semantic annotation and visible-skin review for development must use the `skins` corpus or another eligible development source. Limit review of reserved `other` metadata to provenance and eligibility work that does not inform labels, vocabulary or model tuning; inspect its images or evaluate models only when the [release protocol](release-data-candidates.md) permits it.

### Maintenance throughout the roadmap

- Preserve the locked build, wheel reproducibility check, lint/type gates, and **strictly greater than 80% line and branch coverage** over all maintained production modules. Expand coverage scope when adding modules; do not hide new code or weaken old assertions. Keep GPU experiments separate from ordinary CI.
- Investigate the observed Keras/NumPy warnings using a minimal reproducer and upstream fixes. Any temporary filter should be narrow, documented, and tested; warnings are currently present despite passing tests.
- Update the README to distinguish the original modernization baseline from the recovered local corpus and trained artifacts. Document current commands separately from proposed future commands.
- Preserve the implemented [bounded acquisition](acquisition.md) workflow: retries, response/image validation, exclusive writes and completed/failed/skipped manifests. Unit tests mock the network; no live provider was exercised. Its cooperative request deadline requires an external process cap for a hard wall limit. Additional acquisition remains less urgent than curation and evaluation.
- Review code dependency updates with compatibility checks rather than replacing the already modern toolchain. Track dataset provenance and permissions independently of the repository's code license before publishing images or trained assets.

## Execution and verification plan

Use a rolling-wave plan: specify Phase 1 fully when implementing it, then refine later phases using measured results. For architectural or new-feature work, write the project-required GOTCHA spec before implementation and an ATLAS report before calling the work complete. This roadmap itself does not approve a schema migration or establish a model-quality threshold.

The initial implementation sequence established the following regression obligations; it is retained for traceability, not a request to repeat completed work:

1. **LOCAL-01-T:** Add meaningful failing tests for fresh-process bundle loading, sampler parity, seeded output, invalid metadata, and output collisions. Preserve the existing uniform-sampling tests.
2. **LOCAL-01-I:** Implement the additive bundle/sampler interface after LOCAL-01-T; retain the legacy API path.
3. **LOCAL-01-V:** Run checks and compare fixed latent probes plus generated PNGs against the current local training artifacts. Document numeric tolerances and environment limits.
4. **LOCAL-02-T / LOCAL-02-I:** Separately test, then implement stable manifests and duplicate-group splits; fixtures must exercise ordering, alpha differences, invalid files, and split leakage.
5. **LOCAL-03-T / LOCAL-03-I:** Separately test, then implement dry-run/quarantine/undo behavior on disposable fixtures. Never exercise destructive cleanup against the recovered source corpus.

Parallel work can cover LOCAL-01 and dataset characterization for LOCAL-02, with one integration owner for contracts. Parallel model experiments become useful after LOCAL-05 fixes data and evaluation versions. Do not run concurrent training jobs into a shared checkpoint directory.

The next execution horizon is verification and measured decisions, in this order:

1. Preserve the passing integrated/fresh-wheel gates, recorded environments and qualified isolated phase-memory/continuation evidence. Keep the default GPU parity failure visible; use the tested deterministic compiler profile when its narrower continuation guarantee is required. Expand hardware/architecture evidence only for a declared target or adoption decision.
2. Agree LOCAL-05's draft blinded creator rubric and adoption thresholds before revealing labels. Fix classic/slim and alpha-rendering profiles, collect real reviewer scores, review the report-only similarity policy, and verify actual game behavior separately from renderer diagnostics.
3. Use [completed studies](experiment-results.md), [isolated inference](inference-benchmark.md) and [palette feasibility](palette-study.md) with those judgments to adopt, reject or defer candidates. Retain the baseline on inconclusive evidence; declare a new bounded protocol before any full-corpus escalation or learned-discrete experiment.
4. Assess LOCAL-15 controls and LOCAL-16 compositing usefulness on rendered results. For LOCAL-17, privately review the [existing metadata candidates](annotation-availability.md) and provenance before selecting a vocabulary, annotation protocol or new data source; implement only with verified semantic labels and permitted uses. Preserve the reserved `other` cohort until model, population and eligibility are fixed under the [release protocol](release-data-candidates.md); unresolved release acceptance does not block development work.

Each packet names its input artifact hashes, owned files and output directory before dispatch. Dataset/training contracts have one integration owner; parallel workers use disjoint files and outputs, and GPU studies execute sequentially unless an explicit resource budget supports concurrency. Integrate against the current revision, preserve existing assertions, rerun the applicable gates and request independent review before closing a packet. A failed study does not trigger unbounded retries or silent budget increases.

### Evidence and unresolved decisions

| Kind | Established evidence or remaining question | How to resolve |
| --- | --- | --- |
| Fact | Generation sampler mismatch and implicit GAN budget remain in the compatible legacy path; explicit bundles and budgets address them in the additive package. | Preserve LOCAL-01 and LOCAL-04 contract tests and document the two interfaces. |
| Fact | The current corpus can be decoded and the current model can train on the available GPU. | Retain inventory and recorded training environment; reproduce with the maintained runner. |
| Unknown | Leakage in the historical split and meaningful near-duplicate rate. The new exact-group split is tested; radius-2 scanning found 61,340 candidate pairs, not duplicate prevalence. | Retain manifest/scan evidence and review grouping policy before any new dataset version; do not automatically delete or alter current membership. |
| Unknown | Which change will materially improve blurry outputs. | Paired ablations; no guaranteed architectural remedy. |
| Unknown | Required quality, throughput, and memory targets. | Baseline measurements, supported memory budget, and creator acceptance rubric before release gates. |
| Unknown | Final-test eligibility of the exact-disjoint reserved `other` cohort; historical exposure and head-filter selection bias remain unresolved. | Apply the documented eligibility protocol without tuning on this cohort; obtain independently sampled data if representative population claims require it. |
| Unknown | Whether the recovered PNG metadata candidates provide useful semantic annotations or permission evidence. Presence is measured; meaning and permitted uses are unverified. | Review schemas, vocabulary and provenance privately before conditional-data preparation; obtain eligible labeled data if the candidates are insufficient. |
| Boundary | Existing network downloader and local file/model inputs. | Validate boundary inputs and test failures locally; no network access required for unit tests. |
| Boundary | Optional GPU runtime and implemented local preview renderer. | Retain device smoke checks and verified rendering fixtures; document tested environments and remaining portability limits. |

Historical document-review verification:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  scripts/uv.sh run --locked pytest --no-cov -q
```

Result: **55 passed in 4.79 seconds; 366 deprecation warnings**. Existing coverage and build figures above come from prior recorded verification, not from this command. No model was retrained during the review.

For implementation verification, use the existing [verification script](../scripts/verify.sh) and [reproducibility checker](../scripts/check_reproducibility.py) as documented in the [README](../README.md), plus the new feature-specific checks. The roadmap is complete when its priorities, evidence, dependencies, and acceptance criteria have been reviewed; implementation completion must be established separately for each item.
