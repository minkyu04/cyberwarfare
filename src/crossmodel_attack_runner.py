import argparse
import json
from collections import Counter
from datetime import datetime, timezone


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_CONDITIONS = [
    "C",
    "E",
    "F",
]

AVAILABLE_ATTACK_TYPES = [
    "AT1",
    "AT2",
    "AT3",
    "AT4",
    "AT5",
    "AT6",
    "AT7",
    "AT8",
]

DEFAULT_ATTACK_TYPES = [
    "AT7",
    "AT8",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run selected attack types with C/E/F conditions "
            "using a selectable generator model. "
            "The frozen core experiment runner is not modified."
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
        "--attack-types",
        nargs="+",
        choices=AVAILABLE_ATTACK_TYPES,
        default=DEFAULT_ATTACK_TYPES,
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

    attack_slug = "_".join(
        attack_type.lower()
        for attack_type in args.attack_types
    )

    return (
        f"{model_slug}_"
        f"{args.split}_"
        f"{attack_slug}_"
        "crossmodel"
    )


def write_model_metadata(
    result_dir,
    run_name,
    args,
    original_scenario_count,
    selected_scenario_count,
    attack_counts,
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
        "attack_types": args.attack_types,
        "allow_test": bool(args.allow_test),
        "run_name": run_name,
        "checkpoint_every": int(
            args.checkpoint_every
        ),
        "original_scenario_count": int(
            original_scenario_count
        ),
        "selected_scenario_count": int(
            selected_scenario_count
        ),
        "attack_counts": dict(
            attack_counts
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_override_method": (
            "src.weighted_controlled_rag.MODEL_NAME"
        ),
        "scenario_filter_method": (
            "FullExperimentRunner.scenarios filtered "
            "after initialization by attack_type"
        ),
        "core_runner_modified": False,
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

    from src.full_experiment_runner import (
        FullExperimentRunner,
        RESULT_DIR,
    )

    import src.weighted_controlled_rag as weighted_controlled_rag

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
        "CROSS-MODEL ATTACK-SCOPE EXPERIMENT"
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
        "Attack types:",
        args.attack_types,
    )
    print(
        "Run name:",
        run_name,
    )
    print(
        "=" * 72
    )

    runner = FullExperimentRunner(
        split=args.split,
        conditions=args.conditions,
        smoke=False,
        allow_test=args.allow_test,
        run_name=run_name,
        checkpoint_every=(
            args.checkpoint_every
        ),
    )

    original_scenarios = list(
        runner.scenarios
    )

    selected_attack_types = set(
        args.attack_types
    )

    runner.scenarios = [
        scenario
        for scenario in original_scenarios
        if scenario.get(
            "attack_type"
        ) in selected_attack_types
    ]

    if not runner.scenarios:
        raise RuntimeError(
            "No scenarios matched the requested "
            "attack types."
        )

    attack_counts = Counter(
        scenario.get(
            "attack_type"
        )
        for scenario in runner.scenarios
    )

    print()
    print(
        "Original scenarios:",
        len(original_scenarios),
    )
    print(
        "Selected scenarios:",
        len(runner.scenarios),
    )
    print(
        "Selected attack counts:",
        dict(attack_counts),
    )
    print(
        "Expected condition runs:",
        len(runner.scenarios)
        * len(args.conditions),
    )

    metadata_path = (
        write_model_metadata(
            result_dir=RESULT_DIR,
            run_name=run_name,
            args=args,
            original_scenario_count=(
                len(original_scenarios)
            ),
            selected_scenario_count=(
                len(runner.scenarios)
            ),
            attack_counts=attack_counts,
        )
    )

    runner.run()

    print(
        "\nCross-model metadata:",
        metadata_path,
    )


if __name__ == "__main__":
    main()