"""Installed local workflow; legacy scripts retain their historical defaults."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="skin-gan", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser(
        "acquire", help="Bounded PNG acquisition with per-ID status manifest"
    )
    acquire.add_argument("output", type=Path, help="New output directory")
    acquire.add_argument("--url", required=True, help="Source URL template with {} or {skin_id}")
    acquire.add_argument("--start", type=int, required=True)
    acquire.add_argument("--stop", type=int, required=True, help="Exclusive last ID")
    acquire.add_argument("--provenance", required=True, help="Source/provenance description")
    acquire.add_argument("--timeout", type=float, default=30.0)
    acquire.add_argument("--max-attempts", type=int, default=3)
    acquire.add_argument("--max-workers", type=int, default=4)
    acquire.add_argument("--backoff-seconds", type=float, default=0.5)
    acquire.add_argument("--max-image-bytes", type=int, default=4 * 1024 * 1024)
    prepare = commands.add_parser(
        "prepare", help="Validate PNGs and create a deterministic grouped dataset"
    )
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)
    prepare.add_argument("--seed", type=int, default=1976)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument(
        "--provenance", help="Source/provenance description recorded in the manifest"
    )
    prepare.add_argument("--near-duplicate-distance", type=int, default=0)
    bundle = commands.add_parser("bundle", help="Bundle a saved decoder with encoder latent codes")
    bundle.add_argument("decoder", type=Path)
    bundle.add_argument("codes", type=Path)
    bundle.add_argument("output", type=Path)
    bundle.add_argument("--dataset-fingerprint", required=True)
    bundle.add_argument("--bandwidth", type=float, default=3.16)
    generate = commands.add_parser(
        "generate", help="Seeded generation from a versioned model bundle"
    )
    generate.add_argument("bundle", type=Path)
    generate.add_argument(
        "output", type=Path, help="New directory; existing output is never overwritten"
    )
    generate.add_argument("--count", type=int, default=16)
    generate.add_argument("--seed", type=int, default=1976)
    generate.add_argument(
        "--sampler", choices=("kde", "empirical", "legacy_uniform"), default="kde"
    )
    generate.add_argument("--bandwidth", type=float)
    curate = commands.add_parser(
        "curate", help="Read-only curation plan; --apply explicitly quarantines matches"
    )
    curate.add_argument("source", type=Path)
    curate.add_argument("--policy", choices=("exact", "head-filter"), default="exact")
    curate.add_argument("--apply", type=Path, metavar="QUARANTINE")
    undo = commands.add_parser("undo", help="Restore quarantined files using the checked manifest")
    undo.add_argument("manifest", type=Path)
    evaluate = commands.add_parser(
        "evaluate", help="Development evaluation, not a final generalization claim"
    )
    evaluate.add_argument("bundle", type=Path)
    evaluate.add_argument("data", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--autoencoder", type=Path)
    evaluate.add_argument("--seed", type=int, default=1976)
    evaluate.add_argument("--count", type=int, default=16)
    evaluate.add_argument("--model-type", choices=("classic", "slim"), default="classic")
    experiment = commands.add_parser(
        "experiment", help="Run a validated JSON plan of controlled development experiments"
    )
    experiment.add_argument(
        "plan", type=Path, help="v1 JSON plan; paths are relative to its directory"
    )
    experiment.add_argument("--resume", action="store_true", help="Continue only an identical plan")
    train = commands.add_parser("train", help="Isolated training with explicit phase budgets")
    train.add_argument("data", type=Path)
    train.add_argument("output", type=Path)
    train.add_argument("--seed", type=int, default=1976)
    train.add_argument("--ae-epochs", type=int, default=10)
    train.add_argument("--discriminator-epochs", type=int, default=10)
    train.add_argument("--gan-steps", type=int, default=830)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--encoded-dim", type=int, default=128)
    train.add_argument("--ae-learning-rate", type=float, default=0.001)
    train.add_argument("--discriminator-learning-rate", type=float, default=0.001)
    train.add_argument("--generator-learning-rate", type=float, default=0.00001)
    train.add_argument("--discriminator-updates", type=int, default=1)
    train.add_argument("--checkpoint-interval", type=int, default=50)
    train.add_argument("--bandwidth", type=float, default=3.16)
    train.add_argument("--architecture", choices=("dense", "conv"), default="dense")
    train.add_argument(
        "--reconstruction-loss", choices=("rgba_mse", "visible_rgba"), default="rgba_mse"
    )
    train.add_argument("--device", choices=("cpu", "gpu", "auto"), default="cpu")
    continuation = train.add_mutually_exclusive_group()
    continuation.add_argument("--resume", action="store_true")
    continuation.add_argument(
        "--warm-start", type=Path, help="Prior run directory; reset optimizer, progress and RNG"
    )
    preview = commands.add_parser("preview", help="Local front/back/rotating character preview")
    preview.add_argument("skin", type=Path)
    preview.add_argument("output", type=Path)
    preview.add_argument("--model-type", choices=("classic", "slim"), default="classic")
    preview.add_argument("--alpha-mode", choices=("original", "opaque-base"), default="original")
    gallery = commands.add_parser("gallery", help="Offline batch review and favorite selection")
    gallery.add_argument("source", type=Path)
    gallery.add_argument("output", type=Path)
    gallery.add_argument("--model-type", choices=("classic", "slim"), default="classic")
    gallery.add_argument("--alpha-mode", choices=("original", "opaque-base"), default="original")
    review = commands.add_parser("review", help="Offline anonymous model viewer and score sheet")
    review.add_argument("source", type=Path)
    review.add_argument("output", type=Path)
    review.add_argument("--model-type", choices=("classic", "slim"), default="classic")
    review.add_argument("--alpha-mode", choices=("original", "opaque-base"), default="original")
    review.add_argument("--seed", type=int, default=1976, help="Candidate shuffle seed")
    export = commands.add_parser("export", help="Export original favorites with provenance to ZIP")
    export.add_argument("gallery", type=Path)
    export.add_argument("favorites", type=Path, help="JSON filename list saved by the gallery")
    export.add_argument("output", type=Path)
    palette = commands.add_parser(
        "palette", help="Explicit nearest-color postprocessing with original alpha"
    )
    palette.add_argument("source", type=Path)
    palette.add_argument("output", type=Path)
    palette.add_argument("--colors", required=True, help="Comma-separated #RRGGBB colors")
    preserve = commands.add_parser(
        "preserve", help="Copy exact original pixels in named body parts"
    )
    preserve.add_argument("original", type=Path)
    preserve.add_argument("candidate", type=Path)
    preserve.add_argument("output", type=Path)
    preserve.add_argument(
        "--regions",
        required=True,
        help="Comma-separated head/body/right_arm/left_arm/right_leg/left_leg",
    )
    preserve.add_argument("--model-type", choices=("classic", "slim"), default="classic")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one local command, reporting input/I/O errors without a traceback."""
    args = _parser().parse_args(argv)
    exit_code = 0
    try:
        match args.command:
            case "acquire":
                from minecraft_skin_gan.acquisition import acquire_skins

                manifest_path = acquire_skins(
                    args.url,
                    range(args.start, args.stop),
                    args.output,
                    provenance=args.provenance,
                    timeout=args.timeout,
                    max_attempts=args.max_attempts,
                    max_workers=args.max_workers,
                    backoff_seconds=args.backoff_seconds,
                    max_image_bytes=args.max_image_bytes,
                )
                counts = json.loads(manifest_path.read_text())["counts"]
                result = {"manifest": str(manifest_path), "counts": counts}
                exit_code = 1 if counts["failed"] else 0
            case "prepare":
                from minecraft_skin_gan.dataset import prepare_dataset

                result = prepare_dataset(
                    args.source,
                    args.output,
                    seed=args.seed,
                    validation_fraction=args.validation_fraction,
                    provenance={"description": args.provenance} if args.provenance else None,
                    near_duplicate_distance=args.near_duplicate_distance,
                )
            case "bundle":
                from minecraft_skin_gan.bundle import create_bundle

                result = str(
                    create_bundle(
                        args.decoder,
                        args.codes,
                        args.output,
                        dataset_fingerprint=args.dataset_fingerprint,
                        bandwidth=args.bandwidth,
                    )
                )
            case "generate":
                from minecraft_skin_gan.bundle import generate_bundle

                result = [
                    str(path)
                    for path in generate_bundle(
                        args.bundle,
                        args.output,
                        count=args.count,
                        seed=args.seed,
                        sampler=args.sampler,
                        bandwidth=args.bandwidth,
                    )
                ]
            case "curate":
                from minecraft_skin_gan.dataset import apply_curation, plan_curation

                plan = plan_curation(args.source, policy=args.policy)
                result = str(apply_curation(plan, args.apply)) if args.apply is not None else plan
            case "undo":
                from minecraft_skin_gan.dataset import undo_curation

                result = {"restored": undo_curation(args.manifest)}
            case "evaluate":
                from minecraft_skin_gan.evaluation import evaluate_bundle

                result = str(
                    evaluate_bundle(
                        args.bundle,
                        args.data,
                        args.output,
                        seed=args.seed,
                        count=args.count,
                        autoencoder_path=args.autoencoder,
                        model_type=args.model_type,
                    )
                )
            case "experiment":
                from minecraft_skin_gan.experiments import load_experiment_plan, run_experiments

                plan = load_experiment_plan(args.plan)
                result = str(
                    run_experiments(
                        plan.data_path, plan.output_path, plan.configs, resume=args.resume
                    )
                )
            case "train":
                from minecraft_skin_gan.training import TrainingConfig, run_training

                config = TrainingConfig(
                    data_path=args.data,
                    run_path=args.output,
                    seed=args.seed,
                    encoded_dim=args.encoded_dim,
                    ae_epochs=args.ae_epochs,
                    discriminator_epochs=args.discriminator_epochs,
                    gan_steps=args.gan_steps,
                    batch_size=args.batch_size,
                    device=args.device,
                    ae_learning_rate=args.ae_learning_rate,
                    discriminator_learning_rate=args.discriminator_learning_rate,
                    generator_learning_rate=args.generator_learning_rate,
                    discriminator_updates=args.discriminator_updates,
                    checkpoint_interval=args.checkpoint_interval,
                    bandwidth=args.bandwidth,
                    architecture=args.architecture,
                    reconstruction_loss=args.reconstruction_loss,
                )
                result = str(run_training(config, resume=args.resume, warm_start=args.warm_start))
            case "preview":
                from minecraft_skin_gan.preview import create_preview

                result = str(
                    create_preview(
                        args.skin,
                        args.output,
                        model_type=args.model_type,
                        alpha_mode=args.alpha_mode,
                    )
                )
            case "gallery":
                from minecraft_skin_gan.creator import create_gallery

                result = str(
                    create_gallery(
                        args.source,
                        args.output,
                        model_type=args.model_type,
                        alpha_mode=args.alpha_mode,
                    )
                )
            case "review":
                from minecraft_skin_gan.review import create_review

                result = str(
                    create_review(
                        args.source,
                        args.output,
                        model_type=args.model_type,
                        alpha_mode=args.alpha_mode,
                        seed=args.seed,
                    )
                )
            case "export":
                from minecraft_skin_gan.creator import export_favorites

                selected = json.loads(args.favorites.read_text())
                if not isinstance(selected, list) or any(
                    not isinstance(name, str) for name in selected
                ):
                    raise ValueError("Favorites must be a JSON list of filenames")
                result = str(export_favorites(args.gallery, selected, args.output))
            case "palette":
                from minecraft_skin_gan.creator import recolor_skin

                result = str(recolor_skin(args.source, args.output, args.colors.split(",")))
            case "preserve":
                from minecraft_skin_gan.creator import preserve_regions

                result = str(
                    preserve_regions(
                        args.original,
                        args.candidate,
                        args.output,
                        args.regions.split(","),
                        model_type=args.model_type,
                    )
                )
            case _:
                raise ValueError(f"Unknown command: {args.command}")
    except (OSError, ValueError) as error:
        print(f"skin-gan: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return exit_code
