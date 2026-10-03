"""Controlled experiments validate every candidate before launching training."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from minecraft_skin_gan import experiments
from minecraft_skin_gan.bundle import file_fingerprint
from minecraft_skin_gan.training import TrainingConfig


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "data.npz"
    np.savez(
        path, np.zeros((3, 64, 64, 4), dtype=np.uint8), np.ones((2, 64, 64, 4), dtype=np.uint8)
    )
    return path


def fake_bundle(dataset):
    return SimpleNamespace(
        metadata={"dataset_fingerprint": file_fingerprint(dataset), "bandwidth": 1.0},
        decoder=SimpleNamespace(predict=lambda codes, **kwargs: np.zeros((len(codes), 64, 64, 4))),
    )


def test_sampling_reuses_run_and_same_cohort(dataset, tmp_path, monkeypatch):
    config = TrainingConfig(
        dataset, tmp_path / "train", ae_epochs=1, discriminator_epochs=0, gan_steps=0
    )
    bundle = fake_bundle(dataset)
    monkeypatch.setattr(experiments, "load_bundle", lambda path: bundle)
    sampler = Mock(side_effect=lambda bundle, **kwargs: np.zeros((kwargs["count"], 2)))
    monkeypatch.setattr(experiments, "sample_latents", sampler)
    training = Mock()

    def train(config, **kwargs):
        config.run_path.mkdir()
        (config.run_path / "metrics.json").write_text(
            json.dumps({"completed": {"ae": 1, "discriminator": 0, "gan": 0}})
        )
        (config.run_path / "models").mkdir()
        (config.run_path / "models/autoencoder.keras").write_bytes(b"model")
        return config.run_path / "metrics.json"

    training.side_effect = train
    monkeypatch.setattr(experiments, "run_training", training)
    monkeypatch.setattr(
        experiments,
        "load_decoder",
        lambda path: SimpleNamespace(predict=lambda images, **kwargs: images),
    )
    variants = [
        experiments.ExperimentConfig("base", training=config, sample_count=2),
        experiments.ExperimentConfig(
            "empirical", source_run="base", sampler="empirical", sample_count=2
        ),
    ]
    report = json.loads(
        experiments.run_experiments(dataset, tmp_path / "results", variants).read_text()
    )
    assert training.call_count == 1
    assert report["held_out_indices"] == [0, 1]
    assert report["dataset_sha256"] == file_fingerprint(dataset)
    assert report["results"][0]["reconstruction"]["rgba_mse"] == 0
    assert report["results"][1]["reconstruction"] == report["results"][0]["reconstruction"]
    assert report["results"][1]["generated"]["exact_duplicate_count"] == 1
    assert sampler.call_args.kwargs["sampler"] == "empirical"
    assert (tmp_path / "results/empirical/generated.npy").exists()


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "bad_source",
        "different_count",
        "different_seed",
        "bad_name",
        "bad_budget",
        "run_collision",
        "manifest",
    ],
)
def test_preflight_rejects_whole_matrix_before_training(dataset, tmp_path, monkeypatch, case):
    config = TrainingConfig(dataset, tmp_path / "train", ae_epochs=1, gan_steps=0)
    candidates = [experiments.ExperimentConfig("base", training=config)]
    match = ValueError
    if case == "duplicate":
        candidates.append(candidates[0])
    elif case == "bad_source":
        candidates.append(experiments.ExperimentConfig("other", source_run="absent"))
    elif case == "different_count":
        candidates.append(experiments.ExperimentConfig("other", source_run="base", sample_count=2))
    elif case == "different_seed":
        candidates.append(experiments.ExperimentConfig("other", source_run="base", sample_seed=1))
    elif case == "bad_name":
        candidates.append(experiments.ExperimentConfig("../other", source_run="base"))
    elif case == "bad_budget":
        candidates.append(
            experiments.ExperimentConfig(
                "other", training=replace(config, run_path=tmp_path / "other", gan_steps=-1)
            )
        )
    elif case == "run_collision":
        config.run_path.mkdir()
        match = FileExistsError
    else:
        (dataset.parent / "manifest.json").write_text('{"archive_sha256":"wrong"}')
    trainer = Mock()
    monkeypatch.setattr(experiments, "run_training", trainer)
    with pytest.raises(match):
        experiments.run_experiments(dataset, tmp_path / "results", candidates)
    trainer.assert_not_called()
    assert not (tmp_path / "results").exists()


def test_existing_bundle_identity_and_completed_report_resume(dataset, tmp_path, monkeypatch):
    bundle = fake_bundle(dataset)
    monkeypatch.setattr(experiments, "load_bundle", lambda path: bundle)
    monkeypatch.setattr(
        experiments, "sample_latents", lambda bundle, **kwargs: np.zeros((kwargs["count"], 2))
    )
    config = experiments.ExperimentConfig("sample", bundle_path=tmp_path / "bundle", sample_count=2)
    report = experiments.run_experiments(dataset, tmp_path / "results", [config])
    assert (
        experiments.run_experiments(dataset, tmp_path / "results", [config], resume=True) == report
    )
    with pytest.raises(FileExistsError):
        experiments.run_experiments(dataset, tmp_path / "results", [config])
    with pytest.raises(ValueError):
        experiments.run_experiments(
            dataset, tmp_path / "results", [replace(config, bandwidth=2.0)], resume=True
        )
    bundle.metadata["dataset_fingerprint"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        experiments.run_experiments(dataset, tmp_path / "different", [config])


def test_matrix_varies_one_factor_and_shares_sampling_source(dataset, tmp_path):
    base = TrainingConfig(dataset, tmp_path / "run", gan_steps=2, discriminator_epochs=1)
    configs = experiments.controlled_matrix(
        base, seeds=(1, 2), gan_steps=(0, 2, 3), discriminator_updates=(1, 2), bandwidths=(1.0, 2.0)
    )
    assert len({candidate.name for candidate in configs}) == len(configs)
    training = [candidate for candidate in configs if candidate.training is not None]
    assert {candidate.training.seed for candidate in training} == {1, 2}
    for candidate in training:
        differences = {
            name
            for name in ("gan_steps", "discriminator_updates", "bandwidth")
            if getattr(candidate.training, name) != getattr(base, name)
        }
        assert len(differences) <= 1
    assert any(candidate.sampler == "empirical" and candidate.source_run for candidate in configs)


def test_real_keras_bundle_sampling_and_resume(dataset, tmp_path):
    import os
    import subprocess
    import sys

    from generate_skin import keras
    from minecraft_skin_gan.bundle import create_bundle

    latent = keras.Input(shape=(2,))
    color = keras.layers.Dense(4, activation="sigmoid")(latent)
    decoder = keras.Model(
        latent, keras.layers.Reshape((64, 64, 4))(keras.layers.RepeatVector(4096)(color))
    )
    model = tmp_path / "decoder.keras"
    decoder.save(model)
    codes = tmp_path / "codes.npz"
    np.savez(codes, codes=np.array([[0.0, 0.0], [1.0, 1.0]]), bandwidth=1.0)
    bundle = tmp_path / "bundle"
    create_bundle(model, codes, bundle, dataset_fingerprint=file_fingerprint(dataset))
    variants = [
        experiments.ExperimentConfig("kde", bundle_path=bundle, sample_count=2),
        experiments.ExperimentConfig(
            "empirical", bundle_path=bundle, sampler="empirical", sample_count=2
        ),
    ]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "minecraft-skin-gan.experiment-plan/v1",
                "data_path": "data.npz",
                "output_path": "results",
                "configs": [
                    {"name": "kde", "bundle_path": "bundle", "sample_count": 2},
                    {
                        "name": "empirical",
                        "bundle_path": "bundle",
                        "sampler": "empirical",
                        "sample_count": 2,
                    },
                ],
            }
        )
    )
    completed = subprocess.run(
        [sys.executable, "-m", "minecraft_skin_gan", "experiment", str(plan)],
        cwd=tmp_path,
        env=dict(os.environ, JAX_PLATFORMS="cpu"),
        check=True,
        capture_output=True,
        text=True,
    )
    report_path = tmp_path / "results/experiments.json"
    assert json.loads(completed.stdout) == str(report_path)
    report = json.loads(report_path.read_text())
    assert len(report["results"]) == 2
    assert all(result["generated"]["count"] == 2 for result in report["results"])
    assert all(len(result["nearest_training"]) == 2 for result in report["results"])
    assert (
        experiments.run_experiments(dataset, tmp_path / "results", variants, resume=True)
        == report_path
    )
    np.save(tmp_path / "results/kde/generated.npy", np.zeros((2, 64, 64, 4)))
    with pytest.raises(ValueError, match="output differs"):
        experiments.run_experiments(dataset, tmp_path / "results", variants, resume=True)


@pytest.mark.parametrize(
    "changes",
    [
        dict(sample_count=0),
        dict(sample_count=True),
        dict(sample_seed=-1),
        dict(sampler="unknown"),
        dict(bandwidth=float("nan")),
        dict(bandwidth=True),
    ],
)
def test_sampling_parameters_rejected_before_loading(dataset, tmp_path, monkeypatch, changes):
    loader = Mock()
    monkeypatch.setattr(experiments, "load_bundle", loader)
    candidate = experiments.ExperimentConfig("sample", bundle_path=tmp_path / "bundle", **changes)
    with pytest.raises(ValueError):
        experiments.run_experiments(dataset, tmp_path / "results", [candidate])
    loader.assert_not_called()


def test_empty_matrix_and_report_symlink(dataset, tmp_path):
    with pytest.raises(ValueError):
        experiments.run_experiments(dataset, tmp_path / "results", [])
    (tmp_path / "link").symlink_to(tmp_path / "destination")
    with pytest.raises(FileExistsError):
        experiments.run_experiments(dataset, tmp_path / "link", [])


@pytest.mark.parametrize("case", ["source_count", "data", "budget", "nested", "two_sources"])
def test_bad_training_matrix_does_not_start(dataset, tmp_path, monkeypatch, case):
    config = TrainingConfig(dataset, tmp_path / "run")
    variants = [experiments.ExperimentConfig("base", training=config)]
    if case == "source_count":
        variants.append(experiments.ExperimentConfig("missing"))
    elif case == "data":
        variants.append(
            experiments.ExperimentConfig(
                "other", training=replace(config, data_path=tmp_path / "different.npz")
            )
        )
    elif case == "budget":
        variants[0] = experiments.ExperimentConfig(
            "base", training=replace(config, gan_steps=1000001)
        )
    elif case == "nested":
        variants.append(
            experiments.ExperimentConfig(
                "child", training=replace(config, run_path=config.run_path / "child")
            )
        )
    else:
        variants[0] = experiments.ExperimentConfig("base", training=config, source_run="base")
    trainer = Mock()
    monkeypatch.setattr(experiments, "run_training", trainer)
    with pytest.raises(ValueError):
        experiments.run_experiments(dataset, tmp_path / "results", variants)
    trainer.assert_not_called()


def test_completed_training_config_preflight(dataset, tmp_path, monkeypatch):
    from minecraft_skin_gan.training import _configuration

    config = TrainingConfig(dataset, tmp_path / "run")
    config.run_path.mkdir()
    (config.run_path / "configuration.json").write_text(
        json.dumps({"config": _configuration(config), "dataset_fingerprint": "wrong"})
    )
    trainer = Mock()
    monkeypatch.setattr(experiments, "run_training", trainer)
    with pytest.raises(ValueError, match="configuration or dataset"):
        experiments.run_experiments(
            dataset,
            tmp_path / "results",
            [experiments.ExperimentConfig("base", training=config)],
            resume=True,
        )
    trainer.assert_not_called()


def write_plan(tmp_path, configs):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema": "minecraft-skin-gan.experiment-plan/v1",
                "data_path": "data.npz",
                "output_path": "report",
                "configs": configs,
            }
        )
    )
    return path


def test_json_plan_paths_training_budgets_and_sampling_reference(tmp_path):
    path = write_plan(
        tmp_path,
        [
            {
                "name": "base",
                "training": {
                    "data_path": "data.npz",
                    "run_path": "runs/base",
                    "ae_epochs": 1,
                    "discriminator_epochs": 0,
                    "gan_steps": 0,
                },
            },
            {"name": "empirical", "source_run": "base", "sampler": "empirical"},
        ],
    )
    plan = experiments.load_experiment_plan(path)
    assert plan.data_path == tmp_path / "data.npz"
    assert plan.output_path == tmp_path / "report"
    assert plan.configs[0].training.run_path == tmp_path / "runs/base"
    assert plan.configs[0].training.gan_steps == 0
    assert plan.configs[1].source_run == "base"


@pytest.mark.parametrize(
    "bad",
    [
        {"name": "../unsafe", "bundle_path": "bundle"},
        {"name": 1, "bundle_path": "bundle"},
        {"name": "bad", "bundle_path": 3},
        {"name": "bad", "bundle_path": "bundle", "unknown": 1},
        {"name": "bad", "source_run": "missing"},
        {"name": "bad", "bundle_path": "bundle", "sample_count": True},
        {"name": "bad", "bundle_path": "bundle", "sample_seed": "1"},
        {"name": "bad", "bundle_path": "bundle", "sampler": 2},
        {"name": "bad", "bundle_path": "bundle", "bandwidth": "2"},
        {"name": "bad", "training": {"data_path": "data.npz", "run_path": "run"}},
        {
            "name": "bad",
            "training": {
                "data_path": "data.npz",
                "run_path": "run",
                "ae_epochs": 1,
                "discriminator_epochs": 0,
                "gan_steps": 0,
                "ae_learning_rate": True,
            },
        },
    ],
)
def test_json_plan_invalid_fields_fail_without_mutation(tmp_path, bad):
    path = write_plan(tmp_path, [bad])
    with pytest.raises(ValueError):
        experiments.load_experiment_plan(path)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["plan.json"]


@pytest.mark.parametrize(
    "text",
    [
        '{"schema": 1}',
        '{"schema":"minecraft-skin-gan.experiment-plan/v1","schema":"duplicate"}',
        '{"data_path":NaN}',
        "[]",
        "{",
    ],
)
def test_json_plan_bad_structure_and_nonfinite(tmp_path, text):
    path = tmp_path / "plan.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        experiments.load_experiment_plan(path)


def test_experiment_cli_plan_dispatch_and_errors(tmp_path, monkeypatch, capsys):
    from minecraft_skin_gan.cli import main

    plan = write_plan(tmp_path, [{"name": "sample", "bundle_path": "bundle"}])
    runner = Mock(return_value=tmp_path / "report/experiments.json")
    monkeypatch.setattr(experiments, "run_experiments", runner)
    assert main(["experiment", str(plan), "--resume"]) == 0
    assert runner.call_args.args[:2] == (tmp_path / "data.npz", tmp_path / "report")
    assert runner.call_args.kwargs == {"resume": True}
    assert json.loads(capsys.readouterr().out) == str(tmp_path / "report/experiments.json")
    plan.write_text('{"unknown":1}')
    runner.reset_mock()
    assert main(["experiment", str(plan)]) == 2
    assert "skin-gan: error:" in capsys.readouterr().err
    runner.assert_not_called()


def test_readme_experiment_example_loads(tmp_path):
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "README.md"
    example = (
        readme.read_text()
        .split("### Controlled experiments", 1)[1]
        .split("```json\n", 1)[1]
        .split("```", 1)[0]
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(example)
    plan = experiments.load_experiment_plan(plan_path)
    assert plan.configs[0].training.gan_steps == 0
    assert plan.configs[1].training.gan_steps == 10
    assert plan.configs[2].source_run == "ae-gan"
    assert plan.configs[3].sampler == "empirical"


@pytest.mark.parametrize(
    "architecture,loss,accepted",
    [("dense", "rgba_mse", True), ("conv", "rgba_mse", False), ("dense", "visible_rgba", False)],
)
def test_resume_preflight_migrates_only_historical_defaults(
    dataset, tmp_path, architecture, loss, accepted
):
    from minecraft_skin_gan.training import _configuration

    config = TrainingConfig(
        dataset, tmp_path / "run", architecture=architecture, reconstruction_loss=loss
    )
    legacy = _configuration(config)
    legacy.pop("architecture")
    legacy.pop("reconstruction_loss")
    config.run_path.mkdir()
    (config.run_path / "configuration.json").write_text(
        json.dumps({"config": legacy, "dataset_fingerprint": file_fingerprint(dataset)})
    )
    candidates = [experiments.ExperimentConfig("base", training=config)]
    if accepted:
        identity, *_ = experiments._preflight(
            dataset.resolve(), tmp_path / "results", candidates, True
        )
        assert identity["configs"][0]["training"]["architecture"] == "dense"
    else:
        with pytest.raises(ValueError, match="configuration or dataset"):
            experiments._preflight(dataset.resolve(), tmp_path / "results", candidates, True)
    assert not (tmp_path / "results").exists()
