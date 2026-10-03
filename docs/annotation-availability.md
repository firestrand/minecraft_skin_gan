# Existing annotation and provenance candidates

Audited 2026-10-02 for LOCAL-17. Existing metadata is **present**, but this
availability audit does not verify semantic labels, authorship, source permissions,
or eligibility for conditional training. Do not describe the recovered collection
as wholly devoid of textual metadata, or treat publisher-supplied text as approved
annotations. No image samples were displayed and no models were evaluated.

## Scope and bounded protocol

The audit used the current `dataset-repeat/manifest.json` and the existing
`recovered-other-audit/records.json`, with their exact hashes frozen before launch.
All referenced source pathnames were verified first. Targets were **132,810 valid
`skins` PNGs plus 27,950 `other` PNGs: 160,760 files**. Thirty source files named
`.png` but excluded by the existing strict-content validation were outside scope.
This was the existing recovered project, not a search of other home directories.

The v1 protocol SHA256 is
`1b2d089c3e290196d6684afa26401091506f71d21a3fca53f82fd8c9adfe7e7b`.
Python 3.14.8 and Pillow 12.3.0 inspected PNG textual fields (`tEXt`, `zTXt`,
`iTXt`), `eXIf` presence, and top-level EXIF category availability. Every target
was reopened, identified as a single-frame 64×64 PNG, and decoded with Pillow.
The audit preserved Pillow's default 1 MiB decompressed text-chunk and 64 MiB
aggregate text limits; source files were capped at 16 MiB. No decoder safety
limits were disabled.

The recovered-project sidecar inventory considered CSV, JSON, YAML/YML, TXT,
Markdown, and known license basenames. It excluded generated results, model
directories, development environments, Git metadata, and symlink directories.
Candidate sidecars were bounded to 1 MiB; no archive was extracted. Structured
key categories could be counted without interpreting values or prose.

## Availability counts

| Observation | `skins` | `other` |
| --- | ---: | ---: |
| Strict single-frame 64×64 PNGs audited | 132,810 | 27,950 |
| Files with PNG textual metadata | 1,487 | 77 |
| Files with nonempty string text metadata | 1,487 | 77 |
| Files with EXIF data and top-level tags | 12,144 | 3,196 |
| Image scan or EXIF parse errors | 0 | 0 |

The following counts refer to **files containing a nonempty PNG text field** in
each canonical category. Categories can coexist in one file; do not sum the
rows as distinct annotated examples.

| Canonical text category | `skins` | `other` |
| --- | ---: | ---: |
| Title | 2 | 3 |
| Description | 2 | 3 |
| Author | 3 | 3 |
| Comment | 2 | 0 |
| Software | 113 | 2 |
| Creation Time | 17 | 0 |
| Date | 1 | 0 |
| XMP container | 1,003 | 50 |
| Unknown/noncanonical category | 367 | 23 |

No direct PNG text categories named Label, Tags, Theme, Caption, Keywords,
Source, Copyright, or License were observed under the audit's canonical mapping.
This does **not** establish that such information is absent from XMP, unknown
fields, nested EXIF data, or historical external records. XMP contents and unknown
fields were not semantically interpreted. The later development-only XMP schema
triage below distinguishes XML field categories from these direct PNG text keys.

At the top EXIF level, Description contained nonempty strings in 45 `skins`
files and 3 `other` files; Software did so in 48 and 4 files respectively.
Unknown/noncanonical EXIF categories occurred in all 12,144 and 3,196 EXIF-bearing
files, including 78 and 11 nonempty string tag occurrences. These category counts
identify metadata containers for a separate review, not verified labels. Nested
IFDs and GPS contents were not traversed.

## Development-only XMP schema triage

A separate bounded step examined structured XML field-name categories in all
**1,003 XMP-bearing development `skins` PNGs** identified by the original audit.
It read **zero reserved `other` PNGs** and did not traverse PNG EXIF, nested IFDs,
or GPS data. This is schema availability, not semantic review or label assignment.

Of 1,003 XMP payloads, **1,002 parsed and 1 failed with `ExpatError`**. The failed
payload was retained as a classified error; it was not silently discarded,
repaired, or repeatedly parsed with weaker safeguards. Among successfully parsed
payloads:

| Allowlisted XML field-name category | Files with field | Files with nonempty content |
| --- | ---: | ---: |
| Title | 336 | 335 |
| Description | 799 | 798 |
| Author | 10 | 10 |
| Software | 914 | 914 |
| Tags | 3 | 0 |
| Unknown/noncanonical field bucket | 868 | 868 |

Categories may overlap. The mapping uses allowlisted local field names, so a
Title or Description match does not verify the property's meaning, usefulness,
accuracy, language, or relationship to the visible skin. Nonempty detection is
a boolean; no strings or personal identifiers were retained. Unknown fields
remain unknown, rather than being converted to themes or labels.

Declared or used namespace categories included Dublin Core in 906 successfully
parsed files and XMP Basic in 945; namespace names do not establish semantic
ground truth or rights. No allowlisted Theme, Label, Source, License, or Rights
field-name match was observed **within the successfully parsed payloads**.
This limited observation says nothing about the failed payload, unknown fields,
other metadata containers, or permitted uses.

The v1 schema protocol SHA256 is
`903c79d8c4b6bf6db7c0b7704e1e2e36f933814a5a0945a8995b9721a7d3abea`.
It was frozen before the one run, which finished with exit code **0 in 0.61
seconds** under an external **180-second cap**, with no parser warning categories.
The namespace-aware Expat parser rejects every DTD, entity declaration, and
external entity; no entity or network expansion occurs. XML is bounded to 1 MiB,
depth 64, and 10,000 element/attribute events, with no value tree retained.
Synthetic parser checks verified canonical mapping and DTD refusal before launch.

All 1,003 original source byte hashes matched and were rehashed after the scan.
Reference manifest and prior audit-record hashes remained unchanged. Independent
verification reconciled each category/namespace file count with all 1,003 local
technical records, checked their allowlisted fields, and verified report,
protocol, script, and reference hashes. Error records expose classes only; raw
values, unknown field names, unknown namespace URIs, and private paths are absent
from outputs. No image display, model evaluation, source changes, vocabulary
decision, or additional study followed.

## Original audit filename and sidecar observations

Filename stems were numeric for 132,793 `skins` files and all 27,950 `other`
files. Another 15 `skins` filenames were hexadecimal identifiers and 2 were in
the other-name bucket. No noncanonical filename was recorded in this report or
interpreted as a theme, identity, or annotation.

Only **two sidecars** were found: one recognized project-documentation file and
one recognized project-code-license file. No CSV, JSON, YAML/YML, or TXT candidate
sidecars were found in the scoped recovered project. No annotation schema
categories were extracted from those two files. A code license does not establish
rights to any recovered skin or identify its creator.

## Verification and privacy

The original availability run completed with exit code **0 in 54.14 seconds**, including
the 51.75-second PNG metadata scan, under an external **180-second limit** with
one OMP/BLAS thread. All 160,760 original byte hashes matched existing records
during the scan, then a separate byte-only pass rehashed every source file within
the same cap. Reference records and the frozen protocol remained unchanged.
There were no image, EXIF, or sidecar errors, and no parser warning categories
were recorded.

An independent report check verified artifact hashes, every cohort total and
canonical-category file count against all 160,760 local technical records, and
the allowlisted output fields. The local-only summary contains no private host paths.
No PNG or sidecar metadata values, author names, usernames, emails, raw unknown key names,
unknown sidecar names, or freeform text were persisted or sent in the audit
outputs. Error and warning records contain classes/counts only. Private local
technical records contain cohort/index, byte hashes, and category presence;
they contain no textual values. Source bytes and membership were not changed.

## Consequence for LOCAL-17

The evidence now supports **existing textual/provenance candidates awaiting
review**, rather than an assumption that all metadata is unavailable. It still
does not establish a validated theme/text conditioning dataset. Title,
Description, Author, XMP, and unknown containers may warrant a separately
authorized privacy-conscious schema and provenance review before any annotation
is accepted. Nonempty fields, including any future license-keyword findings, do
not prove ownership or grant training or redistribution permission.

Before conditional training, define the intended labels, source eligibility and
privacy policy, validate annotations against that policy, establish sufficient
coverage and reliable attribution, and freeze an appropriately independent
evaluation cohort. Do not generate labels from numeric filenames, assume a
field's meaning from its name, or manufacture missing labels. Historical
exposure, the `other` cohort's head-filter selection bias, and its measured
near-duplicate links remain separate unresolved questions.

Development annotation review, including checking text against visible images,
must use `skins` or another eligible development source. Reserved `other`
metadata may support non-tuning provenance and eligibility checks; it must not
inform development labels, vocabulary or model choices. Its images and model
evaluation remain subject to the [release protocol](release-data-candidates.md).

## Artifacts

The local-only availability summary (local-only `../artifacts/roadmap-annotation-availability.json`)
contains the frozen protocol, aggregate canonical counts, versions, error counts,
timings, verification, and limitations. Ignored local files under
`images/results/roadmap-2026-10-02/annotation-availability-audit/` retain the hashed
script, protocol, technical category records, full aggregate report, independent
verification, exit status, and privacy-safe launch log. Recovered source files and
local reports are unavailable from a fresh clone. No production code,
dependencies, GPU jobs, network requests, archive extraction, image display, or
model evaluation was introduced.

The local-only development XMP schema summary (local-only `../artifacts/roadmap-annotation-schema.json`)
records category counts and the one classified parse failure. Ignored local
`annotation-schema-audit/` artifacts alongside the availability audit retain its
frozen protocol, script, category-only technical records, aggregate report,
independent verification, log, and terminal status. Reserved `other` source files
were untouched by this schema step.
