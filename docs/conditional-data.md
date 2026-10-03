# Conditional-generation data gate

Reviewed 2026-10-02. LOCAL-17 remains unfinished: the recovered corpus has no
verified semantic label records or source-permission records. Numeric filenames
identify files; they do not establish themes. The reserved `other` cohort must
remain outside development tuning while its release eligibility is unresolved.
Existing metadata candidates are present; availability and verification are
separate questions. Do not assume the corpus has no textual metadata.

## Source evidence

The [privacy-preserving availability audit](annotation-availability.md) checked
all 132,810 valid `skins` and 27,950 `other` PNGs. Text fields were present in
1,487 and 77 files respectively; EXIF was present in 12,144 and 3,196 files.
Small Title, Description, and Author category counts, plus XMP and unknown
containers, identify existing candidates for an authorized schema/provenance
review. No values were retained or interpreted as labels. The scoped recovered
project contained only documentation/code-license sidecars, with no CSV, JSON,
YAML, or TXT annotation candidates. None of these observations establishes
semantic ground truth, authorship, ownership, or source permissions.

A [development-only XMP schema triage](annotation-availability.md#development-only-xmp-schema-triage)
then attempted all 1,003 known `skins` XMP containers: 1,002 parsed and one
`ExpatError` was recorded. Allowlisted Title and Description fields had nonempty
content in 335 and 798 files; three Tags fields were empty. This establishes
structured candidates for a private review, not reviewed semantic labels or
permission evidence. No values were retained, no EXIF/GPS containers were
traversed, and reserved `other` PNGs were not read. Every targeted source byte
hash was rechecked and unchanged.

The legacy [downloader](../download_skins.py) names MinecraftSkins.com as its
default provider. That suggests a possible historical source, but does not prove
where every recovered image came from, who created it, or which permissions
applied when it was collected.

The provider's current [terms](https://www.minecraftskins.com/terms-and-conditions/)
describe user-supplied titles, descriptions and tags. Section 2.1 retains uploader
ownership and grants a license to the site; this is not an observed blanket
training license for this project. Section 5 prohibits automated acquisition or
monitoring of site content. This review fetched policy pages only; it did not
scrape skin pages, collect creator profiles, or download a new corpus. Current
terms do not establish the historical collection conditions.

Even permitted title/tag exports would be candidate annotations, not verified
semantic ground truth. Tags can refer to edits, contests or unrelated concepts.
An annotation review must establish that a chosen label describes the visible
skin rather than relying on substring matching or filenames.

## Required inputs before implementation

First assess whether the recovered metadata candidates can be meaningfully
validated and used with documented permission. Otherwise use a creator-supplied,
permission-documented collection or an explicitly authorized labeled export.
Keep permission evidence private and refer to its
version/fingerprint in the dataset record; do not copy personal data or accounts
into source or tests. For each eligible image, establish:

- Original-byte and decoded-RGBA hashes, dimensions/mode and stable asset ID.
- Source and permitted training/evaluation/redistribution uses, separately.
- Reviewed label vocabulary and annotation version, with ambiguous examples
  explicitly unresolved rather than assigned invented labels.
- Exact-group and reviewed near-duplicate grouping across training and held-out
  conditional evaluation; class counts and coverage of the target population.

The owner must select useful concepts and resolve permitted uses. No collection,
class vocabulary, or numeric minimum per class has been approved. Determine the
pilot's minimum counts, seed set, compute caps and acceptance thresholds before
training; a small convenience set cannot establish general conditional quality.

Use `skins` or another eligible development collection when checking candidate
annotations against visible images. Reserved `other` metadata is limited to
non-tuning provenance and eligibility work; do not use it to select labels,
vocabulary or models, and follow the [release protocol](release-data-candidates.md)
before inspecting its images or evaluating models on them.

## Implementation boundary after the gate

Introduce label metadata and an explicit conditional architecture as a new
versioned experiment. Preserve existing NPZ `arr_0`/`arr_1`, unconditional bundles,
legacy APIs and generation defaults. Test label alignment, missing/unknown-label
errors and fresh-process serialization against the eligible dataset. Evaluate
requested-concept fidelity, rendered quality, diversity and nearest-training
examples on a held-out conditional cohort, alongside the unconditional baseline.

Implementation and adoption require those observed inputs and evaluation results.
An untrained conditional API, automatically invented themes, or metadata scraped
without a permitted collection workflow would not satisfy LOCAL-17.
