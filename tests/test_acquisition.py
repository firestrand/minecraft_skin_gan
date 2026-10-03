"""Acquisition boundaries use local PNG fixtures and mocked transport only."""

import io
import json
from unittest.mock import Mock

import pytest
import requests
from PIL import Image

from minecraft_skin_gan import acquisition


def png(mode="RGBA", size=(64, 64)):
    with Image.new(mode, size) as image, io.BytesIO() as stream:
        image.save(stream, format="PNG")
        return stream.getvalue()


def response(status=200, body=None, headers=None):
    result = Mock()
    result.status_code = status
    result.headers = headers or {}
    result.iter_content.return_value = [png() if body is None else body]
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


def acquire(tmp_path, **kwargs):
    return acquisition.acquire_skins(
        "https://example.test/skins/{}",
        [3, 1],
        tmp_path / "output",
        provenance={"description": "mock fixture"},
        max_workers=1,
        backoff_seconds=0,
        **kwargs,
    )


def test_success_bytes_modes_order_and_resource_cleanup(tmp_path, monkeypatch):
    images = [png("RGB"), png()]
    responses = [response(body=image) for image in images]
    get = Mock(side_effect=responses)
    monkeypatch.setattr(acquisition.requests, "get", get)
    manifest = json.loads(acquire(tmp_path).read_text())
    assert manifest["counts"] == {"completed": 2, "failed": 0, "skipped": 0}
    assert [entry["id"] for entry in manifest["records"]] == [3, 1]
    assert [entry["mode"] for entry in manifest["records"]] == ["RGB", "RGBA"]
    assert all(entry["attempts"] == 1 for entry in manifest["records"])
    assert (tmp_path / "output/3.png").read_bytes() == images[0]
    assert (tmp_path / "output/1.png").read_bytes() == images[1]
    assert sorted(path.name for path in (tmp_path / "output").iterdir()) == [
        "1.png",
        "3.png",
        "manifest.json",
    ]
    for result in responses:
        result.__exit__.assert_called_once()
    assert get.call_args.kwargs == {"stream": True, "timeout": 30.0, "allow_redirects": False}


def test_retry_then_success_and_http_skips(tmp_path, monkeypatch):
    sequence = [response(429), response(503), response(), response(404)]
    monkeypatch.setattr(acquisition.requests, "get", Mock(side_effect=sequence))
    report = json.loads(acquire(tmp_path).read_text())
    assert report["counts"] == {"completed": 1, "failed": 0, "skipped": 1}
    assert report["records"][0]["attempts"] == 3
    assert report["records"][1]["reason"] == "http_404"
    assert not (tmp_path / "output/1.png").exists()
    assert all(item.__exit__.call_count == 1 for item in sequence)


@pytest.mark.parametrize(
    "failure", [requests.Timeout("mock"), requests.ConnectionError("mock"), response(500)]
)
def test_retry_budget_records_failure_without_partial_files(tmp_path, monkeypatch, failure):
    get = Mock(side_effect=[failure] * 4)
    monkeypatch.setattr(acquisition.requests, "get", get)
    report = json.loads(acquire(tmp_path, max_attempts=2).read_text())
    assert get.call_count == 4
    assert report["counts"]["failed"] == 2
    assert all(entry["attempts"] == 2 for entry in report["records"])
    assert list((tmp_path / "output").glob("*.png")) == []


@pytest.mark.parametrize("status,state", [(410, "skipped"), (403, "failed"), (301, "failed")])
def test_nonretryable_statuses(tmp_path, monkeypatch, status, state):
    get = Mock(side_effect=[response(status), response(status)])
    monkeypatch.setattr(acquisition.requests, "get", get)
    report = json.loads(acquire(tmp_path).read_text())
    assert get.call_count == 2
    assert report["counts"][state] == 2


@pytest.mark.parametrize(
    "body,headers,reason",
    [
        (b"not PNG", {}, "invalid_png"),
        (png(size=(32, 64)), {}, "invalid_png"),
        (png(), {"Content-Length": "10000"}, "response_too_large"),
        (b"a" * 129, {}, "response_too_large"),
    ],
)
def test_response_validation_and_cap(tmp_path, monkeypatch, body, headers, reason):
    responses = [response(body=body, headers=headers), response(body=body, headers=headers)]
    monkeypatch.setattr(acquisition.requests, "get", Mock(side_effect=responses))
    report = json.loads(acquire(tmp_path, max_image_bytes=128).read_text())
    assert all(entry["reason"] == reason for entry in report["records"])
    assert all(entry["attempts"] == 1 for entry in report["records"])
    assert list((tmp_path / "output").glob("*.png")) == []


@pytest.mark.parametrize(
    "options",
    [
        {"idsequence": []},
        {"idsequence": [1, 1]},
        {"idsequence": [True]},
        {"idsequence": range(1001)},
        {"max_attempts": 0},
        {"max_workers": 100},
        {"timeout": float("nan")},
        {"backoff_seconds": -1},
        {"max_image_bytes": 0},
        {"skin_url": "https://example.test/no-placeholder"},
        {"skin_url": "https://example.test/{.__class__}"},
        {"skin_url": "https://user:password@example.test/{}"},
        {"provenance": {}},
    ],
)
def test_invalid_configuration_before_mutation(tmp_path, monkeypatch, options):
    get = Mock()
    monkeypatch.setattr(acquisition.requests, "get", get)
    params = {
        "skin_url": "https://example.test/{}",
        "idsequence": [1],
        "output": tmp_path / "output",
        "provenance": "mock fixture",
    }
    params.update(options)
    with pytest.raises(ValueError):
        acquisition.acquire_skins(**params)
    get.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_output_collision_refuses_network_and_preserves_file(tmp_path, monkeypatch):
    get = Mock()
    monkeypatch.setattr(acquisition.requests, "get", get)
    directory = tmp_path / "output"
    directory.mkdir()
    (directory / "3.png").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        acquire(tmp_path)
    assert (directory / "3.png").read_bytes() == b"existing"
    get.assert_not_called()


def test_disk_failure_propagates_and_removes_partial_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(acquisition.requests, "get", Mock(return_value=response()))
    monkeypatch.setattr(acquisition, "_publish", Mock(side_effect=OSError("disk fixture")))
    with pytest.raises(OSError, match="disk fixture"):
        acquire(tmp_path)
    assert list((tmp_path / "output").glob("*.png")) == []
    assert not (tmp_path / "output/manifest.json").exists()


def test_bounded_stream_chunks_and_malformed_length(tmp_path, monkeypatch):
    content = png()
    result = response(headers={"Content-Length": "invalid"})
    result.iter_content.return_value = [b"", content[:30], content[30:]]
    monkeypatch.setattr(acquisition.requests, "get", Mock(return_value=result))
    report = json.loads(acquire(tmp_path, max_image_bytes=len(content)).read_text())
    assert report["counts"]["completed"] == 2
    result.iter_content.assert_called_with(chunk_size=len(content) + 1)


def test_stream_transport_failure_retries_and_discards_partial_bytes(tmp_path, monkeypatch):
    broken = response()

    def chunks(**kwargs):
        yield png()[:30]
        raise requests.exceptions.ChunkedEncodingError("mock truncated stream")

    broken.iter_content.side_effect = chunks
    sequence = [broken, response(), response()]
    monkeypatch.setattr(acquisition.requests, "get", Mock(side_effect=sequence))
    report = json.loads(acquire(tmp_path).read_text())
    assert report["records"][0]["attempts"] == 2
    assert report["records"][0]["attempt_history"][0]["reason"] == "transport_ChunkedEncodingError"
    assert (tmp_path / "output/3.png").read_bytes() == png()
    broken.__exit__.assert_called_once()


def test_fixed_capped_backoff(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acquisition.requests,
        "get",
        Mock(side_effect=[response(429), response(503), response(), response()]),
    )
    sleep = Mock()
    monkeypatch.setattr(acquisition.time, "sleep", sleep)
    acquisition.acquire_skins(
        "https://example.test/{skin_id}",
        [3, 1],
        tmp_path / "output",
        provenance="fixture",
        max_workers=1,
    )
    assert [call.args[0] for call in sleep.call_args_list] == [0.5, 1.0]


@pytest.mark.parametrize("clock,expected_attempts", [([0, 1000], 0), ([0, 0, 1000], 1)])
def test_deadline_before_or_during_body(tmp_path, monkeypatch, clock, expected_attempts):
    get = Mock(return_value=response())
    monkeypatch.setattr(acquisition.requests, "get", get)
    monkeypatch.setattr(acquisition.time, "monotonic", Mock(side_effect=clock))
    result = acquisition._acquire_one(1, "https://example.test/{}", tmp_path, 1.0, 1, 0, 1024)
    assert result["reason"] == "deadline_exceeded"
    assert result["attempts"] == expected_attempts
    assert get.call_count == expected_attempts
    assert list(tmp_path.iterdir()) == []


def test_concurrency_and_input_order(tmp_path, monkeypatch):
    from threading import Barrier

    barrier = Barrier(2, timeout=2)

    def get(url, **kwargs):
        barrier.wait()
        return response()

    monkeypatch.setattr(acquisition.requests, "get", get)
    path = acquisition.acquire_skins(
        "https://example.test/{}", [8, 2], tmp_path / "output", provenance="fixture", max_workers=2
    )
    report = json.loads(path.read_text())
    assert [record["id"] for record in report["records"]] == [8, 2]
    assert report["counts"]["completed"] == 2


def test_publication_collision_cannot_replace_another_file(tmp_path, monkeypatch):
    def get(*args, **kwargs):
        (tmp_path / "output/3.png").write_bytes(b"other writer")
        return response()

    monkeypatch.setattr(acquisition.requests, "get", get)
    with pytest.raises(FileExistsError):
        acquire(tmp_path)
    assert (tmp_path / "output/3.png").read_bytes() == b"other writer"
    assert not list((tmp_path / "output").glob(".acquire-*"))


def test_real_staging_disk_failure_cleans_temporary_bytes(tmp_path, monkeypatch):
    from pathlib import Path

    original = Path.write_bytes

    def fail_write(path, data):
        original(path, data[:5])
        raise OSError("mock partial disk write")

    monkeypatch.setattr(acquisition.requests, "get", Mock(return_value=response()))
    monkeypatch.setattr(Path, "write_bytes", fail_write)
    with pytest.raises(OSError, match="partial disk write"):
        acquire(tmp_path)
    assert list((tmp_path / "output").iterdir()) == []


def test_manifest_disk_failure_keeps_completed_images_without_false_report(tmp_path, monkeypatch):
    publish = acquisition._publish

    def fail_manifest(path, data):
        if path.name == "manifest.json":
            raise OSError("mock manifest failure")
        publish(path, data)

    responses = [response(), response()]
    monkeypatch.setattr(acquisition.requests, "get", Mock(side_effect=responses))
    monkeypatch.setattr(acquisition, "_publish", fail_manifest)
    with pytest.raises(OSError, match="manifest failure"):
        acquire(tmp_path)
    assert sorted(path.name for path in (tmp_path / "output").iterdir()) == ["1.png", "3.png"]
    assert all(item.__exit__.call_count == 1 for item in responses)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"idsequence": "123"},
        {"idsequence": [-1]},
        {"timeout": True},
        {"max_attempts": True},
        {"max_image_bytes": 16777217},
        {"skin_url": "file:///{}"},
        {"skin_url": "https://example.test/{skin_id!r}"},
        {"skin_url": "https://example.test/{}/{}"},
        {"skin_url": "https://example.test/{"},
        {"provenance": ""},
        {"provenance": {"description": float("nan")}},
        {"provenance": {"unsupported": object()}},
        {"provenance": {1: "invalid"}},
        {"skin_url": "https://example.test:invalid/{}"},
        {"skin_url": "https://example.test:0/{}"},
    ],
)
def test_more_invalid_options_never_mutate(tmp_path, monkeypatch, kwargs):
    get = Mock()
    monkeypatch.setattr(acquisition.requests, "get", get)
    options = {
        "skin_url": "https://example.test/{}",
        "idsequence": [1],
        "output": tmp_path / "output",
        "provenance": "fixture",
    }
    options.update(kwargs)
    with pytest.raises(ValueError):
        acquisition.acquire_skins(**options)
    get.assert_not_called()
    assert not (tmp_path / "output").exists()
