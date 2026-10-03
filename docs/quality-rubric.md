# Development skin-quality rubric

Status: draft, 2026-10-02. This protocol separates rendered quality from reconstruction metrics. The owner has not yet approved weights or adoption thresholds; no human scores have been collected.

Use the same held-out cohort and candidate sample counts. Render each PNG with the explicit classic/slim model type, showing front, back and overlays at identical scale and lighting. Include nearest-neighbor enlarged atlas views; do not smooth pixels. Shuffle candidate labels with a recorded seed and hide architecture, loss, bandwidth and training metrics until scoring is complete. Keep the label mapping separately from score sheets. Rate both independently sampled batches and paired common-latent samples where encoder space is shared.

Record one alpha-rendering profile for every comparison. `original` preserves source alpha in all preview layers; `opaque-base` renders base faces opaque, matching the documented reference-renderer material, while preserving overlay alpha and exported PNG bytes. Use the same profile for all candidates in a scored comparison. A separate profile comparison may help assess alpha effects, but neither mode establishes actual game acceptance.

| Criterion | 1: poor | 3: mixed | 5: strong |
| --- | --- | --- | --- |
| Detail | Blurred or unintelligible facial/clothing detail | Some readable features, some blurred regions | Crisp, readable pixel detail at native resolution |
| Body coherence | Disconnected or misplaced body features | Recognizable parts with visible inconsistencies | Head, limbs and clothing form a consistent character |
| Palette | Uncontrolled color noise | Plausible colors with distracting patches | Deliberate-looking, consistent color relationships |
| Seams | Distracting discontinuities in rendered geometry | Some joins distract | Joins look intentional or unobtrusive |
| Wearability | Reviewer would discard it | Would use after substantial editing | Would use with minor or no editing |
| Batch variety | Candidates mostly repeat one design | Several distinct candidates | Broad useful variety without sacrificing coherence |

Collect per-skin scores for the first five criteria and one batch score for variety, plus an accept/edit/discard choice and brief reason. Record reviewer identity by local pseudonym, review date, artifact hashes, sample count and completion count. Do not replace missing scores with averages or invent reviewer votes. Report score distributions and preferences; a small single-reviewer pilot does not support population claims. The owner chooses practical adoption thresholds before labels are revealed.

File checks (64×64 RGBA, finite values and PNG roundtrip) are mechanical. [Renderer diagnostics](skin-format.md) describe base/overlay alpha and cuboid-edge continuity; they do not certify Minecraft upload acceptance. Hidden-RGB changes and pixel-distance comparisons must not be presented as wearability evidence. Nearest-training similarity prompts copying review, not an automatic originality verdict.

Release evaluation additionally requires provenance-reviewed, genuinely unexposed disjoint data and a selected candidate fixed before reviewing that data. Development scores from the recovered corpus remain development evidence.

## Prepared development review packet

A local [draft gallery](../images/results/roadmap-2026-10-02/creator-review-draft/gallery/index.html)
contains 32 anonymously named candidates from the existing AE-only and AE+GAN
KDE-3.16 batches. Both used the same sixteen latent probes and sampling seed 1976;
candidate order was shuffled with seed 20261002. Classic geometry and the original
alpha profile are fixed. Exported RGBA bytes and gallery copies were checked.

The [blank score sheet](../images/results/roadmap-2026-10-02/creator-review-draft/scores.csv)
has one row per candidate. Complete the five per-skin criteria and accept/edit/discard
choice only after agreeing thresholds. Batch variety is assessed after scoring,
with the two batches grouped by the separately stored label mapping. Keep that
mapping closed until individual scores and thresholds are fixed; the preparing
agent knows the mapping and cannot supply blinded human votes.

This is a single-seed development pilot using previously inspected generated
outputs, not release evidence. No scores have been collected. The gallery and
score sheet depend on local trained artifacts and are unavailable from a fresh
clone. Packet evidence (local-only `../artifacts/roadmap-creator-review-draft.json`) records
protocol and hashes without revealing candidate assignments.

## Interactive scoring viewer

The [local 32-candidate model viewer](../images/results/roadmap-2026-10-02/scoring-viewer/index.html) provides anonymous navigation, rotation/layer/zoom controls, the five ratings, accept/edit/discard and notes. [Viewer instructions](scoring-viewer.md) explain browser drafts and CSV/JSON backups. This is a separately fingerprinted reshuffle; use its researcher-only candidate map to join exported scores to the original packet after scoring. Keep source mappings sealed until thresholds and scores are fixed. No human ratings have been collected; browser tests used a disposable copy only.
