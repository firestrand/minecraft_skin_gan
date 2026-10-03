"""Behavioral contracts for skin preprocessing and bounded downloads."""

import runpy
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

import download_skins
import remove_duplicates
import sort_skins


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def test_legacy_dataset_schema_and_split(tmp_path, monkeypatch):
    spec = find_spec("create_skins_array")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin)
    monkeypatch.chdir(tmp_path)
    # Filesystem order is deliberately retained; the historical split is positional.
    for value in range(10):
        write_png(tmp_path / f"images/skins/{value}.png", np.full((64, 64, 4), value, np.uint8))
    runpy.run_path(str(source), run_name="__main__")
    with np.load(tmp_path / "images/train_test.npz") as data:
        assert data.files == ["arr_0", "arr_1"]
        assert data["arr_0"].shape == (8, 64, 64, 4)
        assert data["arr_1"].shape == (2, 64, 64, 4)
        from sklearn.model_selection import train_test_split

        original = np.stack(
            [
                np.full((64, 64, 4), int(p.stem), np.uint8)
                for p in Path("images/skins").glob("*.png")
            ]
        )
        expected_train, expected_test = train_test_split(original, test_size=0.2, random_state=1976)
        np.testing.assert_array_equal(data["arr_0"], expected_train)
        np.testing.assert_array_equal(data["arr_1"], expected_test)


def test_legacy_sort_keeps_one_uniform_channel(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Path("images/other").mkdir(parents=True)
    variable = np.indices((64, 64)).sum(axis=0).astype(np.uint8) * 4
    keep = np.stack(
        [variable, variable, np.zeros_like(variable), np.full_like(variable, 255)], axis=-1
    )
    move = np.zeros((64, 64, 4), np.uint8)
    write_png(Path("images/skins/keep.png"), keep)
    write_png(Path("images/skins/move.png"), move)
    sort_skins.main()
    assert Path("images/skins/keep.png").exists()
    assert not Path("images/skins/move.png").exists()
    with Image.open("images/other/move.png") as image:
        np.testing.assert_array_equal(np.asarray(image), move)
    assert capsys.readouterr().out == "done\n"


def test_legacy_duplicate_uses_perceptual_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pattern = np.random.default_rng(1976).integers(0, 256, (64, 64, 4), dtype=np.uint8)
    write_png(Path("images/first.png"), pattern)
    write_png(Path("images/copy.png"), pattern)
    write_png(Path("images/different.png"), np.zeros_like(pattern))
    original_order = list(Path("images").glob("*.png"))
    remove_duplicates.main()
    retained_pattern = [p for p in original_order if p.stem != "different"][0]
    assert retained_pattern.exists()
    assert len(list(Path("images").glob("*.png"))) == 2
    assert Path("images/different.png").exists()


def test_create_import_has_no_processing(tmp_path, monkeypatch):
    spec = find_spec("create_skins_array")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        Image, "open", MagicMock(side_effect=AssertionError("import opened an image"))
    )
    namespace = runpy.run_path(str(source))
    assert callable(namespace["main"])
    assert list(tmp_path.iterdir()) == []


def test_dataset_converts_rgb_to_rgba(tmp_path):
    import create_skins_array

    for value in range(5):
        write_png(tmp_path / f"skins/{value}.png", np.full((64, 64, 3), value, np.uint8))
    create_skins_array.main(tmp_path / "skins", tmp_path / "dataset.npz")
    with np.load(tmp_path / "dataset.npz") as data:
        for key in data.files:
            assert data[key].dtype == np.uint8
            assert np.all(data[key][..., 3] == 255)


@pytest.mark.parametrize("case", ["empty", "one", "mismatched", "corrupt"])
def test_dataset_invalid_inputs_raise_without_archive(tmp_path, case):
    import create_skins_array

    if case != "empty":
        write_png(tmp_path / "skins/one.png", np.zeros((64, 64, 4), np.uint8))
    if case == "mismatched":
        write_png(tmp_path / "skins/two.png", np.zeros((32, 64, 4), np.uint8))
    if case == "corrupt":
        (tmp_path / "skins/broken.png").write_bytes(b"invalid PNG")
    with pytest.raises((ValueError, OSError)):
        create_skins_array.main(tmp_path / "skins", tmp_path / "dataset.npz")
    assert not (tmp_path / "dataset.npz").exists()


@pytest.mark.parametrize("low_variance_channels", [0, 1, 2, 3])
def test_filter_strict_threshold_and_ignores_alpha(low_variance_channels):
    array = np.zeros((64, 64, 4), np.uint8)
    # Standard deviation is exactly 10 for 0/20, and 9 for 0/18.
    for channel in range(3):
        array[8:16:2, :32, channel] = 18 if channel < low_variance_channels else 20
    assert sort_skins.should_filter_skin(array) is (low_variance_channels >= 2)
    array[..., 3] = np.random.default_rng(1).integers(0, 256, (64, 64), dtype=np.uint8)
    assert sort_skins.should_filter_skin(array) is (low_variance_channels >= 2)


def test_sort_creates_destination_and_leaves_non_png(tmp_path):
    source = tmp_path / "skins"
    source.mkdir()
    write_png(source / "skin.png", np.zeros((64, 64, 4), np.uint8))
    original = (source / "skin.png").read_bytes()
    (source / "notes.txt").write_text("keep me", encoding="utf-8")
    sort_skins.main(source, tmp_path / "other")
    assert (tmp_path / "other/skin.png").read_bytes() == original
    assert (source / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_remove_duplicates_empty_directory(tmp_path):
    remove_duplicates.main(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_remove_duplicates_keeps_rgb_rgba_hash_equivalence(tmp_path):
    pattern = np.random.default_rng(44).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    write_png(tmp_path / "rgb.png", pattern)
    write_png(tmp_path / "rgba.png", np.dstack((pattern, np.full((64, 64), 255, np.uint8))))
    (tmp_path / "ignored.txt").write_text("keep", encoding="utf-8")
    remove_duplicates.main(tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert (tmp_path / "ignored.txt").exists()


@pytest.mark.parametrize("status", [200, 404, 500])
def test_download_positional_api_and_status(tmp_path, monkeypatch, status):
    monkeypatch.chdir(tmp_path)
    response = MagicMock(status_code=status, content=b"raw response bytes")
    response.__enter__.return_value = response
    get = MagicMock(return_value=response)
    monkeypatch.setattr(download_skins.requests, "get", get)
    assert download_skins.download_skin("https://example.test/{}", 123) is (status == 200)
    get.assert_called_once_with("https://example.test/123", stream=True, timeout=30.0)
    response.__exit__.assert_called_once()
    if status == 200:
        assert Path("images/skins/123.png").read_bytes() == b"raw response bytes"
    else:
        assert not Path("images").exists()


def test_download_transport_error_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_skins.requests,
        "get",
        MagicMock(side_effect=download_skins.requests.Timeout("deadline")),
    )
    with pytest.raises(download_skins.requests.Timeout, match="deadline"):
        download_skins.download_skin("https://example.test/{}", 1, tmp_path / "skins", timeout=0.01)
    assert not (tmp_path / "skins").exists()


def test_download_write_failure_closes_response(tmp_path, monkeypatch):
    response = MagicMock(status_code=200, content=b"bytes")
    response.__enter__.return_value = response
    monkeypatch.setattr(download_skins.requests, "get", MagicMock(return_value=response))
    (tmp_path / "1.png").mkdir()
    with pytest.raises(IsADirectoryError):
        download_skins.download_skin("https://example.test/{}", 1, tmp_path)
    response.__exit__.assert_called_once()


@pytest.mark.parametrize("start,stop", [(5, 108), (9, 10), (4, 4), (8, 2)])
def test_download_main_bounds_range_and_counts_success(tmp_path, monkeypatch, start, stop):
    download = MagicMock(side_effect=lambda _url, skin_id, _out, **_kwargs: skin_id % 2 == 0)
    bar = MagicMock()
    bar.__enter__.return_value = bar
    bar_factory = MagicMock(return_value=bar)
    monkeypatch.setattr(download_skins, "download_skin", download)
    monkeypatch.setattr(download_skins, "Bar", bar_factory)
    download_skins.main(
        "https://example.test/{}", start, stop, tmp_path, max_workers=2, batch_size=100, timeout=2.0
    )
    assert sorted(call.args[1] for call in download.call_args_list) == list(range(start, stop))
    assert all(
        call.args[2] == tmp_path and call.kwargs == {"timeout": 2.0}
        for call in download.call_args_list
    )
    assert bar.next.call_count == sum(skin_id % 2 == 0 for skin_id in range(start, stop))
    bar_factory.assert_called_once_with("Processing", max=max(0, stop - start))
    bar.__exit__.assert_called_once()


def test_download_main_worker_error_propagates_and_closes_bar(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download_skins, "download_skin", MagicMock(side_effect=OSError("disk error"))
    )
    bar = MagicMock()
    bar.__enter__.return_value = bar
    monkeypatch.setattr(download_skins, "Bar", MagicMock(return_value=bar))
    with pytest.raises(OSError, match="disk error"):
        download_skins.main(start=1, stop=2, output_dir=tmp_path)
    bar.__exit__.assert_called_once()


@pytest.mark.parametrize("kwargs", [{"max_workers": 0}, {"batch_size": 0}])
def test_download_main_invalid_concurrency_raises(kwargs):
    with pytest.raises(ValueError):
        download_skins.main(start=1, stop=2, **kwargs)
