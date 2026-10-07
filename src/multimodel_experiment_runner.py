import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_CONDITIONS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the full synthetic experiment with a selectable "
            "generator model without modifying the core runner."
        )
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face causal language model name. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "test",
        ],
        default="dev",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=DEFAULT_CONDITIONS,
        default=DEFAULT_CONDITIONS,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    parser.add_argument(
        "--allow-test",
        action="store_true",
    )

    parser.add_argument(
        "--run-name",
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )

    return parser.parse_args()


def sanitize_model_name(model_name):
    return (
        model_name
        .replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
    )


def resolve_run_name(args):
    if args.run_name:
        return args.run_name

    model_slug = sanitize_model_name(
        args.model_name
    )

    suffix = (
        "smoke"
        if args.smoke
        else args.split
    )

    return (
        f"{model_slug}_{suffix}"
    )


def write_model_metadata(
    result_dir,
    run_name,
    args,
):
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        result_dir
        / f"{run_name}_model_metadata.json"
    )

    metadata = {
        "model_name": args.model_name,
        "split": args.split,
        "conditions": args.conditions,
        "smoke": bool(args.smoke),
        "allow_test": bool(args.allow_test),
        "run_name": run_name,
        "checkpoint_every": int(
            args.checkpoint_every
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_override_method": (
            "src.weighted_controlled_rag.MODEL_NAME"
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return metadata_path


def main():
    args = parse_args()

    # Importing the existing full runner first preserves its
    # dataset-activation order. No experiment instance is created
    # during import.
    from src.full_experiment_runner import (
        FullExperimentRunner,
        RESULT_DIR,
    )
    import src.weighted_controlled_rag as weighted_controlled_rag

    # WeightedControlledRAG.__init__ reads MODEL_NAME at runtime,
    # so changing the module-level value here selects the generator
    # while leaving the core experiment files unchanged.
    weighted_controlled_rag.MODEL_NAME = (
        args.model_name
    )

    run_name = resolve_run_name(
        args
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "MULTI-MODEL EXPERIMENT WRAPPER"
    )
    print(
        "=" * 72
    )
    print(
        "Generator model:",
        args.model_name,
    )
    print(
        "Split:",
        args.split,
    )
    print(
        "Conditions:",
        args.conditions,
    )
    print(
        "Smoke:",
        args.smoke,
    )
    print(
        "Run name:",
        run_name,
    )
    print(
        "=" * 72
    )

    metadata_path = (
        write_model_metadata(
            result_dir=RESULT_DIR,
            run_name=run_name,
            args=args,
        )
    )

    runner = FullExperimentRunner(
        split=args.split,
        conditions=args.conditions,
        smoke=args.smoke,
        allow_test=args.allow_test,
        run_name=run_name,
        checkpoint_every=(
            args.checkpoint_every
        ),
    )

    runner.run()

    print(
        "\nModel metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()
