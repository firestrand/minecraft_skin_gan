"""Bounded future acquisition with validated PNGs and auditable outcomes.

This additive interface leaves the legacy downloader unchanged. It does not
establish permission to obtain, train on, or redistribute any provider's images.
"""

import hashlib
import io
import json
import math
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Formatter
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError


class _Rejected(Exception):
    """A permanent response failure, suitable for an acquisition record."""


def _integer(value: Any, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def _number(value: Any, name: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be finite between {minimum} and {maximum}")


def _configuration(
    skin_url: str,
    idsequence: Sequence[int],
    provenance: Mapping[str, Any] | str,
    timeout: float,
    max_attempts: int,
    max_workers: int,
    backoff_seconds: float,
    max_image_bytes: int,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if isinstance(idsequence, (str, bytes)) or not isinstance(idsequence, Sequence):
        raise ValueError("IDs require a sequence of between 1 and 1000 unique integers")
    try:
        count = len(idsequence)
    except OverflowError as error:
        raise ValueError("IDs require a sequence of between 1 and 1000 unique integers") from error
    if not 1 <= count <= 1000:
        raise ValueError("IDs require a sequence of between 1 and 1000 unique integers")
    ids = tuple(idsequence)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63
        for value in ids
    ) or len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique nonnegative integers below 2**63")
    _number(timeout, "timeout", 0.001, 60)
    _integer(max_attempts, "max_attempts", 5)
    _integer(max_workers, "max_workers", 16)
    _number(backoff_seconds, "backoff_seconds", 0, 10)
    _integer(max_image_bytes, "max_image_bytes", 16 * 1024 * 1024)
    if not isinstance(skin_url, str) or len(skin_url) > 2048:
        raise ValueError("URL template must be a string below 2049 characters")
    parsed = list(Formatter().parse(skin_url))
    placeholders = [
        (field, spec, conversion) for _, field, spec, conversion in parsed if field is not None
    ]
    if len(placeholders) != 1 or placeholders[0] not in [("", "", None), ("skin_id", "", None)]:
        raise ValueError("URL template must contain exactly one {} or {skin_id} placeholder")
    url = urlsplit(skin_url.format(0, skin_id=0))
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.fragment
        or url.port == 0
    ):
        raise ValueError("URL template requires HTTP(S), a host, and no credentials or fragment")
    if isinstance(provenance, str):
        if not provenance.strip():
            raise ValueError("Provenance description must be nonempty")
        origin: dict[str, Any] = {"description": provenance}
    elif isinstance(provenance, Mapping) and provenance:
        if any(not isinstance(key, str) for key in provenance):
            raise ValueError("Provenance keys must be strings")
        origin = dict(provenance)
    else:
        raise ValueError("Provenance requires a nonempty description or mapping")
    try:
        serialized = json.dumps(origin, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Provenance must contain finite JSON values") from error
    if len(serialized.encode("utf-8")) > 16 * 1024:
        raise ValueError("Provenance exceeds 16 KiB")
    return ids, json.loads(serialized)


def _publish(output: Path, data: bytes) -> None:
    """Publish complete bytes exclusively using a same-filesystem hard link."""
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=output.parent) as directory:
        temporary = Path(directory) / "image"
        temporary.write_bytes(data)
        output.hardlink_to(temporary)


def _body(response: requests.Response, maximum: int, deadline: float) -> bytes:
    length = response.headers.get("Content-Length")
    if length is not None:
        try:
            if int(length) > maximum:
                raise _Rejected("response_too_large")
        except ValueError:
            pass  # Streaming still enforces the cap when the header is malformed.
    data = bytearray()
    for chunk in response.iter_content(chunk_size=min(65536, maximum + 1)):
        if time.monotonic() >= deadline:
            raise _Rejected("deadline_exceeded")
        if len(data) + len(chunk) > maximum:
            raise _Rejected("response_too_large")
        data.extend(chunk)
    return bytes(data)


def _image(data: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format != "PNG"
                or image.size != (64, 64)
                or getattr(image, "n_frames", 1) != 1
            ):
                raise _Rejected("invalid_png")
            image.load()
            mode = image.mode
            with image.convert("RGBA") as rgba:
                fingerprint = hashlib.sha256(rgba.tobytes()).hexdigest()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise _Rejected("invalid_png") from error
    return {
        "mode": mode,
        "shape": [64, 64, 4],
        "bytes": len(data),
        "byte_sha256": hashlib.sha256(data).hexdigest(),
        "rgba_sha256": fingerprint,
    }


def _acquire_one(
    skin_id: int,
    skin_url: str,
    output: Path,
    timeout: float,
    max_attempts: int,
    backoff_seconds: float,
    max_image_bytes: int,
) -> dict[str, Any]:
    delays = [min(30.0, backoff_seconds * 2**index) for index in range(max_attempts - 1)]
    deadline = time.monotonic() + max_attempts * 2 * timeout + sum(delays)
    record: dict[str, Any] = {
        "id": skin_id,
        "status": "failed",
        "attempts": 0,
        "reason": "deadline_exceeded",
        "attempt_history": [],
    }
    url = skin_url.format(skin_id, skin_id=skin_id)
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            record["reason"] = "deadline_exceeded"
            break
        record["attempts"] = attempt
        retry = False
        try:
            with requests.get(
                url, stream=True, timeout=min(timeout, remaining), allow_redirects=False
            ) as response:
                status = response.status_code
                record["http_status"] = status
                if status == 200:
                    data = _body(response, max_image_bytes, deadline)
                    info = _image(data)
                else:
                    record["reason"] = f"http_{status}"
                    record["status"] = "skipped" if status in {404, 410} else "failed"
                    retry = status == 429 or 500 <= status <= 599
                    data, info = None, {}
        except requests.RequestException as error:
            record["reason"] = f"transport_{type(error).__name__}"
            record.pop("http_status", None)
            retry = True
            data, info = None, {}
        except _Rejected as error:
            record["reason"] = str(error)
            data, info = None, {}
        if data is not None:
            # Disk/publication errors remain exceptions, separate from network outcomes.
            path = output / f"{skin_id}.png"
            _publish(path, data)
            record.update(info, status="completed", reason="validated_png", path=path.name)
        record["attempt_history"].append(
            {
                "attempt": attempt,
                "reason": record["reason"],
                **({"http_status": record["http_status"]} if "http_status" in record else {}),
            }
        )
        if not retry or attempt == max_attempts:
            return record
        delay = min(delays[attempt - 1], max(0, deadline - time.monotonic()))
        if delay:
            time.sleep(delay)
    return record


def acquire_skins(
    skin_url: str,
    idsequence: Sequence[int],
    output: Path | str,
    *,
    provenance: Mapping[str, Any] | str,
    timeout: float = 30.0,
    max_attempts: int = 3,
    max_workers: int = 4,
    backoff_seconds: float = 0.5,
    max_image_bytes: int = 4 * 1024 * 1024,
) -> Path:
    """Acquire explicit IDs into a new directory; return its ordered JSON manifest.

    HTTP/transport/content outcomes are recorded per ID. Disk errors propagate:
    already completed PNGs may remain, and no completed manifest is fabricated.
    Redirects are not followed. Only trusted explicit HTTP(S) templates are used.
    Request timeout is connect/read inactivity. The per-ID deadline is checked
    cooperatively between requests/chunks; it cannot interrupt a blocked read.
    """
    ids, origin = _configuration(
        skin_url,
        idsequence,
        provenance,
        timeout,
        max_attempts,
        max_workers,
        backoff_seconds,
        max_image_bytes,
    )
    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)

    def worker(skin_id: int) -> dict[str, Any]:
        return _acquire_one(
            skin_id, skin_url, destination, timeout, max_attempts, backoff_seconds, max_image_bytes
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        records = list(executor.map(worker, ids, buffersize=max_workers))
    manifest = {
        "schema": "minecraft-skin-gan.acquisition/v1",
        "configuration": {
            "url_template": skin_url,
            "ids": list(ids),
            "timeout": timeout,
            "timeout_semantics": "connect/read inactivity; cooperative deadline cannot interrupt blocked transport",
            "max_attempts": max_attempts,
            "max_workers": max_workers,
            "backoff_seconds": backoff_seconds,
            "max_image_bytes": max_image_bytes,
            "redirect_policy": "refuse",
            "cooperative_deadline_seconds_per_id": max_attempts * 2 * timeout
            + sum(min(30.0, backoff_seconds * 2**index) for index in range(max_attempts - 1)),
        },
        "provenance": origin,
        "counts": {
            state: sum(record["status"] == state for record in records)
            for state in ("completed", "failed", "skipped")
        },
        "records": records,
        "permissions": "unverified; caller-provided provenance is not evidence of redistribution rights or semantic labels",
    }
    path = destination / "manifest.json"
    _publish(
        path,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"),
    )
    return path
