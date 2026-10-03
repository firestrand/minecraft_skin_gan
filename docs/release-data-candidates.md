# Reserved release-data candidates

Reviewed 2026-10-02. A read-only structural/hash audit found an existing candidate
cohort in the recovered archive's `images/other` directory. Preserve this cohort
without model evaluation, visual inspection, curation, or automatic membership
changes until a candidate model, evaluation rubric, and eligibility protocol are
fixed. No image samples were viewed during this audit.

## Observed evidence

The source was
`~/Downloads/minecraft_skin_gan_extracted_nrq9341_/minecraft_skin_gan/images/other`.
The reference was the current `dataset-repeat/manifest.json`, covering all 132,810
valid skins across training and validation. Byte and filename comparisons also
included its thirty invalid source records, for 132,840 reference filenames.

| Check | Result |
| --- | ---: |
| PNG files audited | 27,950 |
| Strict valid single-frame 64×64 PNGs | 27,950 |
| Invalid files | 0 |
| Unique decoded RGBA groups | 27,950 |
| Unique original-byte groups | 27,950 |
| Exact decoded RGBA overlap with reference training | 0 |
| Exact decoded RGBA overlap with reference validation | 0 |
| Exact byte overlap with all reference records | 0 |
| Filename overlap with all reference records | 0 |

Original modes were RGBA (27,884), palette/P (58), RGB (7), and grayscale/L (1).
Every file was identified as PNG, checked for 64×64 dimensions and a single frame,
verified with Pillow, then reopened, decoded, and converted to RGBA solely to hash
its decoded bytes. Original bytes were independently hashed. No samples, rendered
sheets, model outputs, or semantic image judgments were involved.

The audit took 14.74 seconds under an external 180-second process cap. A separate
read-only pass rechecked all 27,950 source byte hashes plus the reference manifest
and local record hashes. Source content remained unchanged. A separate bounded
nonzero-distance cross-reference is recorded below; exact independence alone does
not establish near-duplicate, shared-author, or shared-source independence.

## Nonzero-distance cross-reference

Before launch, the audit froze a v1 protocol with SHA256
`64d792222e15380045cb7d9c0621a485ae4940104915938a739855a320f1ec6a`.
It compared **all 27,950 `other` files against all 132,810 valid current skins**,
including 106,248 training and 26,562 development-validation files. The thirty
invalid reference records were excluded. No model outputs or selected sample
cohort were used as the reference.

The algorithm matches dataset preparation: `ImageHash.phash` on decoded RGBA,
`hash_size=8`, `highfreq_factor=4`, with 64-bit Hamming distance at most **2**.
It derives luminance from RGB and ignores alpha. Exact hash buckets and all
2,081 zero-, one-, and two-bit XOR masks per distinct query hash enumerate the
radius completely, without a quadratic image comparison or materializing every
candidate pair. Both corpora happened to contain distinct pHashes for every file.

| Cross-reference observation | Count |
| --- | ---: |
| Candidate file pairs, both reference splits | 16,364 |
| Candidate pairs against current training | 12,881 |
| Candidate pairs against current development validation | 3,483 |
| Candidate pairs at distance 0 / 1 / 2 | 0 / 0 / 16,364 |
| Distinct `other` files with any candidate link | 3,303 |
| Distinct current files with any candidate link | 8,177 |
| Distinct `other` files linked to training / validation | 2,910 / 1,506 |
| Distinct current training / validation files linked | 6,483 / 1,694 |
| Deterministic candidate pairs retained in the local report | 1,000 |

These are candidate counts, **not population prevalence or verified semantic
equivalence**. A file may link to both reference splits, so the two `other`
split-specific counts overlap. The retained pairs include relative filenames,
split, byte and RGBA group hashes, both pHashes, and distances; the list is a
deterministic prefix rather than a representative review sample. No pairs were
viewed or judged. Hidden RGB, ignored alpha, recoloring, and the limitations of a
64-bit RGB-derived hash can affect these links; this radius does not establish
independence at larger distances or shared-source independence.

The single scan completed with exit code 0 in **8.66 seconds**, under the frozen
external **180-second cap**, using one OMP/BLAS thread. It matched all prior source
byte and RGBA hashes while computing technical metadata, then independently
rehashed all 27,950 original source files. The source inventory, existing records,
and reference manifest identities remained unchanged. A separate metadata-only
verification checked all 1,000 retained pair distances, names, hashes, and
uniqueness against original records, and compared the index with brute-force
Hamming comparisons for 32 evenly spaced queries against every reference hash.
No source pixels, membership, or split assignments changed.

## Exposure and selection limits

The current maintained models read the prepared `skins` archives, including
training and validation data, rather than these `other` files. This audit found no
exact-content overlap with those reference groups, but the radius-2 cross-reference
found candidate links from 3,303 `other` files to current data. Exact-disjointness
is evidence against direct exact-example leakage; it does not establish that
variants of current examples are absent or prove historical unexposedness. Earlier
models, manual inspection, derived files, and prior use of the recovered archive
remain unknown.

The legacy [filter](../sort_skins.py) moves a skin into `other` when more than one
RGB channel has standard deviation below 10 in rows 8–15 and columns 0–31. This
is selection by head-color variation, not random population sampling. The
recovered folder is therefore a potentially biased filter-selected source, and
the audit does not verify that every file arrived through that exact historical
operation. Do not describe this cohort as a representative wearable-skin sample.

Filename IDs do not supply semantic theme/text labels. Neither the archive nor
the code license establishes source permissions, training rights, or redistribution
rights. Those questions remain separate from structural validity and hash overlap.

## Conditional use

This is a plausible **reserved stress-test cohort**, or a narrowly defined final
cohort for the filter-selected distribution, conditional on eligibility checks.
The repository should no longer assume there is no local exact-disjoint data
candidate. It also should not declare representative final-test data established.
The entire exact-disjoint collection cannot yet be counted as a verified
leakage-independent final set. If the eventual eligibility protocol groups
cross-reference variants with current examples, its decisions must resolve the
candidate links to **both training and exposed development validation** before
freezing final membership. A radius-2 link is a review flag, not an automatic
exclusion or an eligibility verdict; this audit assigns neither.

Before final evaluation:

1. Document source provenance and permitted uses, and investigate historical
   exposure without opening image samples for candidate tuning.
2. Fix a near-duplicate/shared-source eligibility policy and resolve the recorded
   radius-2 candidates under it before measuring model quality. Define any further
   checks or exclusions in advance, retain their decision records, and preserve
   the recovered source without automatic moves or exclusions.
3. Fix the candidate model, quality rubric, numeric gates, and the target population.
   Explicitly distinguish head-filter stress results from representative population
   claims; obtain a separately sampled cohort if those claims require one.
4. Reserve a versioned, fingerprinted cohort under that protocol and evaluate it
   once for the selected candidate. Do not use it to select architecture, sampler,
   palette settings, stopping budgets, or quality thresholds.

No cohort was split, excluded, assigned release eligibility, or evaluated by this
audit. Semantic labels and verified provenance remain unresolved.

## Artifacts

The local-only summary (local-only `../artifacts/roadmap-other-corpus-audit.json`) records protocol,
counts, modes, reference hashes, exposure limits, and verification. The ignored
local [full report](../images/results/roadmap-2026-10-02/recovered-other-audit/audit.json)
and [per-file metadata](../images/results/roadmap-2026-10-02/recovered-other-audit/records.json)
retain all hash and overlap records. Local artifacts and recovered images are not
available from a fresh clone.

The local-only radius-2 summary (local-only `../artifacts/roadmap-other-near-duplicate-audit.json`)
records the frozen protocol, complete counts, source verification, and independent
checks. Ignored local artifacts in
`images/results/roadmap-2026-10-02/recovered-other-near-audit/` retain `protocol.json`,
the hashed `run_audit.py`, per-file pHashes, the full report with its 1,000 candidate
pairs, independent verification, exit status, and launch log. No production code,
dependencies, training jobs, network requests, model evaluation, visual inspection,
or source-image writes were introduced.
