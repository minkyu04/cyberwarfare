import argparse
import json
from datetime import datetime, timezone

import pandas as pd


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ABLATIONS = [
    "baseline",
    "no-history",
    "no-weight",
    "all-block",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run F-condition ablation experiments for the "
            "stateful weighted minimum-loss disclosure controller."
        )
    )

    parser.add_argument(
        "--ablation",
        choices=ABLATIONS,
        required=True,
        help=(
            "baseline: original F; "
            "no-history: ignore prior disclosed facts; "
            "no-weight: use unit weights only during removal optimization; "
            "all-block: block every candidate protected fact."
        ),
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face generator model. "
            f"Default: {DEFAULT_MODEL_NAME}"
        ),
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
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


def sanitize_name(value):
    return (
        str(value)
        .replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
    )


def resolve_run_name(args):
    if args.run_name:
        return args.run_name

    model_slug = sanitize_name(
        args.model_name
    )

    scope = (
        "smoke"
        if args.smoke
        else args.split
    )

    return (
        f"ablation_{args.ablation}_"
        f"{model_slug}_{scope}"
    )


def apply_no_history(runner):
    """
    Ignore previously disclosed facts during controller evaluation.

    The experiment runner's independent observed-response history
    remains unchanged, so cross-turn leakage is still measurable.
    """

    def empty_exposure_history(
        session_id,
        user_id,
    ):
        del session_id
        del user_id
        return []

    runner.rag.session_manager.get_exposed_fact_ids = (
        empty_exposure_history
    )


def apply_no_weight(runner):
    """
    Use equal utility weights only while the controller chooses
    facts to remove.

    The original utility map is restored immediately after each
    evaluate_turn() call. This is necessary because the evaluation
    metrics use the same controller object to calculate business
    utility retention. Keeping unit weights after control would
    contaminate the evaluation metric itself.
    """

    controller = runner.controller
    original_evaluate_turn = (
        controller.evaluate_turn
    )

    original_utility_map = dict(
        controller.utility_map
    )

    unit_utility_map = {
        fact_id: 1.0
        for fact_id
        in original_utility_map
    }

    def evaluate_turn_without_weights(
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None,
    ):
        controller.utility_map = (
            unit_utility_map
        )

        try:
            return original_evaluate_turn(
                user_id=user_id,
                exposed_facts=exposed_facts,
                candidate_facts=candidate_facts,
                as_of=as_of,
            )

        finally:
            controller.utility_map = (
                original_utility_map
            )

    controller.evaluate_turn = (
        evaluate_turn_without_weights
    )

    runner.rag.controller.evaluate_turn = (
        evaluate_turn_without_weights
    )


def apply_all_block(runner):
    """
    Remove every protected candidate fact instead of selecting
    the minimum-loss subset.
    """

    def block_all_candidates(
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None,
    ):
        del user_id
        del exposed_facts
        del as_of

        candidates = list(
            dict.fromkeys(
                candidate_facts
            )
        )

        return {
            "removed_facts": candidates,
            "allowed_facts": [],
        }

    runner.controller.evaluate_turn = (
        block_all_candidates
    )

    runner.rag.controller.evaluate_turn = (
        block_all_candidates
    )


def apply_ablation(
    runner,
    ablation,
):
    if ablation == "baseline":
        return

    if ablation == "no-history":
        apply_no_history(
            runner
        )
        return

    if ablation == "no-weight":
        apply_no_weight(
            runner
        )
        return

    if ablation == "all-block":
        apply_all_block(
            runner
        )
        return

    raise ValueError(
        f"Unknown ablation: {ablation}"
    )


def append_experiment_columns(
    csv_path,
    ablation,
    model_name,
):
    if not csv_path.exists():
        return False

    df = pd.read_csv(
        csv_path
    )

    if "ablation" in df.columns:
        df["ablation"] = (
            ablation
        )
    else:
        df.insert(
            0,
            "ablation",
            ablation,
        )

    if "model_name" in df.columns:
        df["model_name"] = (
            model_name
        )
    else:
        insert_index = (
            1
            if "ablation" in df.columns
            else 0
        )

        df.insert(
            insert_index,
            "model_name",
            model_name,
        )

    df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    return True


def annotate_result_files(
    result_dir,
    run_name,
    ablation,
    model_name,
):
    paths = [
        result_dir
        / f"{run_name}_results.csv",
        result_dir
        / f"{run_name}_summary.csv",
        result_dir
        / f"{run_name}_attack_summary.csv",
    ]

    annotated = []

    for path in paths:
        if append_experiment_columns(
            csv_path=path,
            ablation=ablation,
            model_name=model_name,
        ):
            annotated.append(
                str(path)
            )

    return annotated


def write_metadata(
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
        / (
            f"{run_name}_"
            "ablation_metadata.json"
        )
    )

    descriptions = {
        "baseline": (
            "Original F condition with session history, "
            "business-value weighting, and selective "
            "minimum-loss removal."
        ),
        "no-history": (
            "Prior disclosure history is ignored during "
            "controller evaluation."
        ),
        "no-weight": (
            "All candidate facts receive unit utility only "
            "during minimum-loss removal optimization. "
            "Evaluation retains the original business values."
        ),
        "all-block": (
            "Every protected candidate fact is removed "
            "instead of applying selective minimum-loss control."
        ),
    }

    metadata = {
        "ablation":
            args.ablation,
        "description":
            descriptions[
                args.ablation
            ],
        "model_name":
            args.model_name,
        "split":
            args.split,
        "smoke":
            bool(
                args.smoke
            ),
        "allow_test":
            bool(
                args.allow_test
            ),
        "run_name":
            run_name,
        "checkpoint_every":
            int(
                args.checkpoint_every
            ),
        "condition":
            "F",
        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
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
        "F ABLATION EXPERIMENT"
    )
    print(
        "=" * 72
    )
    print(
        "Ablation:",
        args.ablation,
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

    runner = FullExperimentRunner(
        split=args.split,
        conditions=[
            "F"
        ],
        smoke=args.smoke,
        allow_test=args.allow_test,
        run_name=run_name,
        checkpoint_every=(
            args.checkpoint_every
        ),
    )

    apply_ablation(
        runner=runner,
        ablation=args.ablation,
    )

    metadata_path = (
        write_metadata(
            result_dir=RESULT_DIR,
            run_name=run_name,
            args=args,
        )
    )

    runner.run()

    annotated = (
        annotate_result_files(
            result_dir=RESULT_DIR,
            run_name=run_name,
            ablation=args.ablation,
            model_name=args.model_name,
        )
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "ABLATION RUN COMPLETE"
    )
    print(
        "=" * 72
    )
    print(
        "Metadata:",
        metadata_path,
    )

    for path in annotated:
        print(
            "Annotated:",
            path,
        )


if __name__ == "__main__":
    main()
