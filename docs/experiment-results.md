# Development experiment results

Recorded: 2026-10-02. These are local development studies, not release or generalization claims. Original images and models remain unchanged. The concise sampling evidence (local-only `../artifacts/roadmap-sampling-study.json`) records artifact hashes; large arrays and sheets are local ignored outputs under `images/results/roadmap-2026-10-02/`.

## Paired full-corpus sampling study

The study reused the selected pre-GAN autoencoder and the final decoder from the same maintained run: 10 AE epochs, 10 discriminator epochs, 830 GAN updates. The encoder and latent-code data were unchanged. Each candidate used 16 samples and seed 1976. KDE bandwidths were 0.25, 1.0 and 3.16; empirical sampling was a fourth candidate per decoder. No candidate retrained a model. The development archive SHA256 was `1f97438e56695cac13898ac67f420c96f62ff3f3ba7df3ff7d49c21c3d290ac8`.

| Decoder / sampler | Mean pairwise RGBA MSE | Mean nearest-training RGBA MSE |
| --- | ---: | ---: |
| AE / KDE 0.25 | 0.14039 | 0.01505 |
| AE / KDE 1.0 | 0.13957 | 0.01771 |
| AE / KDE 3.16 | 0.14419 | 0.03710 |
| AE / empirical | 0.10373 | 0.01329 |
| GAN / KDE 0.25 | 0.14053 | 0.01416 |
| GAN / KDE 1.0 | 0.13942 | 0.01630 |
| GAN / KDE 3.16 | 0.14158 | 0.03321 |
| GAN / empirical | 0.10309 | 0.01221 |

All eight batches contained 16 distinct quantized images. Narrower bandwidths brought outputs closer to training examples; empirical sampling reduced within-batch pixel variation in this probe. Neither quantity measures useful variety or proves copying/originality. Sampling modes now have measured effects, but there is no demonstrated superior mode for wearable quality.

The paired reconstruction cohort comprised the same validation rows 0–15:

| Metric | Selected AE-only | AE+GAN |
| --- | ---: | ---: |
| RGBA MSE | 0.026720 | 0.025717 |
| Visible RGB MSE | 0.051755 | 0.050906 |
| Alpha MAE | 0.039702 | 0.034858 |

The final decoder had slightly lower errors on this fixed cohort. This small one-seed comparison does not establish visual improvement or causation across training runs. Inspection of the GAN/KDE-3.16 sheet still showed blurred detail. Human wearability scores, game validation and final-test evaluation remain unverified. Retain the existing baseline and explicit sampler choices pending the [blinded rubric](quality-rubric.md).

## Architecture and loss pilot

A declared nine-run pilot compared dense/RGBA-MSE, convolutional/RGBA-MSE and dense/visible-RGBA-loss autoencoders with seeds 1976, 2026 and 2027. The real-image subset contains 4,096 training and 1,024 validation images selected without replacement inside the existing splits, with membership and source hashes recorded. Each run has five AE epochs and zero discriminator/GAN updates, so this isolates the autoencoder changes. Model selection uses the same full 1,024-image validation set; paired report reconstructions use its first 16 rows. Sampling seeds/counts are fixed, but independently learned latent spaces do not imply shared concepts.

Aggregate wall budget: 30 minutes for nine sequential runs; host/device planning budgets: 16/4 GiB. All nine runs completed. Summed measured training time was 118.46 seconds; aggregate process-lifetime high-water memory stayed below 4.79 GiB host / 2.55 GiB device. See the measured pilot (local-only `../artifacts/roadmap-model-pilot.json`) for per-seed metrics and parameter counts.

| Candidate | Full validation RGBA MSE, mean (range over three seeds) | AE parameters |
| --- | ---: | ---: |
| Dense / RGBA MSE | 0.042015 (0.041598–0.042351) | 35,044,512 |
| Conv / RGBA MSE | 0.040710 (0.039381–0.041848) | 1,136,612 |
| Dense / visible RGBA | 0.162690 (0.159380–0.167662) | 35,044,512 |

Convolutional validation RGBA error was lower in two seeds and slightly higher in one; visible RGB error on the 16-image report cohort was higher in every seed. It is a substantially smaller candidate worth further study, not an adopted replacement. The alpha-aware objective reduced visible RGB error on that cohort in all three seeds, while its all-RGBA error was much higher. Hidden RGB is intentionally excluded from its objective; the report does not treat that metric divergence as unexplained improvement or silently change the selection metric.

Timing and memory observations include compilation/cache reuse and process-lifetime allocator counters across sequential runs. They cannot establish isolated architecture latency or memory differences. A subsequent [isolated inference benchmark](inference-benchmark.md) measured faster conv prediction with higher host/device allocations for its sixteen-image batch; it does not establish isolated training peaks. No human scores or game acceptance tests establish wearability. Recommendation: **defer adoption**, retain both explicit optional variants, and perform rendered review plus required training-resource measurements before a full-corpus escalation. A failure would stop execution and be retained; none occurred in this pilot.

## Near-duplicate scan limits

A read-only full-corpus pHash scan at Hamming radius 4 hit its declared 180-second limit (exit 124), producing no completed candidate report. Radius 2 completed in 90.14 seconds under a 120-second cap, finding **61,340 cross-split candidate pairs**, with the first 1,000 retained. This is a candidate-pair count, not duplicate prevalence or unique images. See scan evidence (local-only `../artifacts/roadmap-near-duplicate-review.json`).

Agent visual inspection and exact RGBA probes of the first eight candidates found several obvious design variants and pairs differing in only 60 and 17 texels; one pHash pair was visibly dissimilar. This deterministic prefix is not a representative sample or a human review. RGB-derived pHash ignores alpha. The development split contains similarity candidates and cannot be described as similarity-disjoint. Keep grouping **report-only** until a reviewed similarity policy supports any new split; preserve this existing immutable dataset version. No source files or split memberships changed.

## GAN duration and update-ratio pilot

Nine sequential runs used the same real 4096/1024 subset, seeds 1976/2026/2027, five dense AE epochs and two discriminator epochs. One-factor contrasts compared 128 vs 256 GAN updates, and one vs two discriminator updates per generator update at 256 updates. Learning rates, batch 128, sample 16 / seed 1976 and validation rows were fixed. The declared 15-minute limit and 16/8 GiB host/device budgets were respected; summed training time was 309.03 seconds, process-lifetime peaks 8.06/6.90 GiB. Hashed evidence (local-only `../artifacts/roadmap-gan-pilot.json`) records per-seed results.

| Candidate | Mean full-validation RGBA MSE | Mean visible RGB MSE (16 rows) | Mean alpha MAE (16 rows) | Mean within-batch RGBA variation |
| --- | ---: | ---: | ---: | ---: |
| 256 updates / ratio 1 | 0.044586 | 0.080616 | 0.088631 | 0.098025 |
| 128 updates / ratio 1 | 0.043045 | 0.077765 | 0.086047 | 0.090471 |
| 256 updates / ratio 2 | 0.045254 | 0.082573 | 0.085183 | 0.099966 |

Shorter GAN training had lower average reconstruction errors with less pixel variation. Doubling discriminator updates improved average alpha error slightly but worsened RGB/RGBA reconstruction. These are developmental trade-offs; no result establishes better rendered quality, copying safety or a universal duration/ratio. Do not silently replace the retained legacy/default policy. Human review and unexposed evaluation remain necessary before adoption.

## Palette-aware representation

The [bounded palette study](palette-study.md) measured 8/16/32 colors on the same exported generated batch. It preserved exact alpha/hidden RGB and all 16 unique images, but posterized colors without restoring missing spatial detail. This completes a small palette-aware feasibility investigation with a defer recommendation; it does not implement or disprove a learned VQ model.

Its follow-up nearest-training review used all sixteen exports per representation
and the entire training split. Mean distances changed slightly in both directions;
this supplies quantitative copying-review flags rather than a copying-safety verdict.

## Isolated architecture training resources

The architecture resource gap has separate [matched training evidence](spatial-training-resources.md):
six cold fresh processes, dense/conv across the same three seeds and real pilot
cohort, with five AE epochs each. Mean whole-process RSS was 3.210 GiB dense and
2.669 GiB conv; mean whole-command wall time was 36.79 and 25.15 seconds. Conv
had higher phase-end device in-use/pool peaks (0.990/1.998 GiB versus
0.726/1.002 GiB). Those measurements include initialization, compilation and
exports and cover AE-only training. They do not supply rendered quality,
adversarial-training resources or a full-corpus adoption decision. Historical
pilot numeric differences in both directions are retained in the report.

## Reserved release-data candidate

A read-only [archive audit](release-data-candidates.md) found 27,950 strict PNGs in
`images/other`, with no exact RGBA overlap against current training or validation.
No images were visually inspected or model-evaluated. Keep this cohort reserved:
historical exposure, provenance, near-duplicate independence and head-filter
selection bias remain unresolved. It does not yet establish representative or
genuinely unexposed release evaluation.

The subsequent radius-2 RGB-pHash cross-reference found 16,364 candidate pairs,
linking 3,303 reserved files with 8,177 current files, including both training and
exposed validation. Counts are candidate links, not confirmed duplicates or copying.
No files were excluded or assigned release eligibility. The audit evidence (local-only `../artifacts/roadmap-other-near-duplicate-audit.json`)
and release protocol record the narrower independence claim.
