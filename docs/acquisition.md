# Bounded acquisition for future data needs

The additive `minecraft_skin_gan.acquisition.acquire_skins` interface supports
explicit, small acquisition batches when more data is needed. The legacy
`download_skins.py` behavior and tests remain unchanged. Implementation and
verification used mocked transport and local image fixtures; no new external
dataset was downloaded.

## Interface

```python
from pathlib import Path
from minecraft_skin_gan.acquisition import acquire_skins

manifest = acquire_skins(
    "https://example.test/skins/{}",  # Replace with an authorized provider template.
    range(100, 110),
    Path("new-acquisition"),
    provenance={"description": "Record source and acquisition context"},
    timeout=30.0,
    max_attempts=3,
    max_workers=4,
    backoff_seconds=0.5,
    max_image_bytes=4 * 1024 * 1024,
)
```

IDs are explicit and retain their supplied order in the manifest. The URL template
requires exactly one `{}` or `{skin_id}` placeholder and an HTTP(S) host; credentials,
fragments, format conversions, and arbitrary attribute/index substitutions are
rejected. Provide a nonempty provenance description or JSON-compatible mapping.
Keep secrets out of URLs and provenance because both are recorded in the manifest.

The installed command accepts a half-open ID range:

```bash
timeout 120s scripts/uv.sh run --locked skin-gan acquire /path/to/new-acquisition \
  --url 'https://example.test/skins/{}' --start 100 --stop 110 \
  --provenance 'Source and acquisition context' \
  --timeout 30 --max-attempts 3 --max-workers 4 \
  --backoff-seconds 0.5 --max-image-bytes 4194304
```

`example.test` is a documentation placeholder. The command returns a manifest path
and counts: exit 0 for completed/skipped outcomes, 1 when any ID failed, and 2 for
invalid input or filesystem errors. External `timeout` returns its own status when
it terminates the process.

## Limits and outcomes

| Input | Accepted range |
| --- | --- |
| IDs | 1–1,000 unique nonnegative integer IDs, each below `2**63` |
| Timeout | 0.001–60 seconds |
| Attempts per ID | 1–5 |
| Workers | 1–16 |
| Initial backoff | 0–10 seconds; exponential retry delays capped at 30 seconds |
| Response byte cap | 1 byte–16 MiB; default 4 MiB |
| Provenance JSON | At most 16 KiB |

The worker count and executor buffer bound concurrency and queued submissions.
Responses stream in bounded chunks; an excessive `Content-Length` is rejected
early, and actual streamed bytes are capped regardless of the header. Malformed
length headers do not bypass that cap. Redirects are refused and recorded as
failed HTTP outcomes rather than followed to another endpoint.

- **Completed:** HTTP 200 with a single-frame, decodable 64×64 PNG convertible to
  RGBA. Preserve original file bytes and original mode, and record byte and decoded
  RGBA SHA256 hashes. RGB PNGs are accepted without rewriting them.
- **Skipped:** HTTP 404 or 410, with no retry or output image.
- **Failed:** other HTTP statuses, exhausted request/transport retries, oversized
  responses, invalid image content, or an observed cooperative deadline expiry.
  Retry only HTTP 429, HTTP 5xx, and `requests` transport exceptions; permanent
  content and other HTTP failures are recorded immediately.

Every network/content outcome records ID, status, attempts, reason, and attempt
history. Records are deterministically ordered by the supplied IDs, independent of
worker completion order. `manifest.json` uses schema
`minecraft-skin-gan.acquisition/v1` and includes resolved acquisition options,
caller provenance, and `counts.completed`, `counts.failed`, and `counts.skipped`.
Elapsed wall time and provider response order are not claimed reproducible.

The requests timeout measures connect/read **inactivity**, not total wall time.
A cooperative per-ID deadline is derived from twice the timeout per allowed
attempt plus the capped backoff delays. It is checked before attempts and between
stream chunks and recorded as `cooperative_deadline_seconds_per_id`. It cannot
interrupt a blocked read, DNS lookup, or a peer trickling data while a chunk is
being assembled. Use an external process timeout when a hard wall-clock limit is
required; the API makes no hard cancellation guarantee.

## Publication and failure handling

The output directory must be new. Validation happens before creating it or making
requests, and an existing file, directory, or symlink is refused. Each image and
the final manifest are written to temporary files on the destination filesystem,
then published exclusively through hard links. A publication collision never
replaces another file; temporary bytes are removed on ordinary exceptions.
Atomic visibility does not promise crash durability.

Filesystem failures propagate rather than being mislabeled as network failures or
completed acquisitions. Already published images from other IDs may remain; a
complete manifest is not fabricated after a failed write. Forced termination can
likewise leave completed images without a final manifest, or temporary files.
There is no automatic resume into an existing directory. Inspect the partial
directory and start any subsequent acquisition in a separate new directory.

## Verification and boundaries

Focused fixture tests cover retry exhaustion and backoff, transport errors during
streaming, permanent HTTP outcomes, response caps, PNG decoding, original bytes
and modes, ordering under concurrent workers, response cleanup, cooperative
deadline observations, invalid configuration, exclusive collisions, and failed
staging writes. No test requests a live provider.

This capability prepares for future acquisition; it does not resolve source
permissions, provider rate-policy requirements, genuinely unexposed test data, or
semantic theme/text labels. Caller-supplied provenance is recorded evidence to
review, not proof of redistribution rights.
