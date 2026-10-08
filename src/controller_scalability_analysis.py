import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import pandas as pd

from src.utility_sensitivity_analysis import (
    load_rules,
    load_utility_map,
    validate_utility_coverage,
    build_rule_components,
    solve_minimum_hitting_set,
    safety_preserved,
)


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_RULE_FILE = (
    BASE_DIR
    / "data"
    / "full_experiment"
    / "inference_rules.json"
)

DEFAULT_UTILITY_FILE = (
    BASE_DIR
    / "data"
    / "full_experiment"
    / "fact_utility.json"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "controller_scalability"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Controller-only scalability benchmark for the "
            "exact minimum-loss disclosure optimization kernel."
        )
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
        ],
        help="Synthetic candidate-fact counts.",
    )

    parser.add_argument(
        "--instances",
        type=int,
        default=20,
        help=(
            "Number of independently generated rule graphs "
            "per candidate-fact count."
        ),
    )

    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=10,
        help=(
            "Number of solver timing repetitions per "
            "generated instance."
        ),
    )

    parser.add_argument(
        "--actual-repeats",
        type=int,
        default=100,
        help=(
            "Timing repetitions for each connected component "
            "of the final experiment rule set."
        ),
    )

    parser.add_argument(
        "--rule-density",
        type=float,
        default=1.5,
        help=(
            "Synthetic rule count as approximately "
            "candidate_fact_count * rule_density."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rule-file",
        type=Path,
        default=DEFAULT_RULE_FILE,
    )

    parser.add_argument(
        "--utility-file",
        type=Path,
        default=DEFAULT_UTILITY_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def percentile(values, q):
    if not values:
        return float("nan")

    values = sorted(
        float(value)
        for value in values
    )

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * q
    )

    lower = math.floor(
        position
    )

    upper = math.ceil(
        position
    )

    if lower == upper:
        return values[
            lower
        ]

    fraction = (
        position
        - lower
    )

    return (
        values[lower]
        * (
            1.0
            - fraction
        )
        + values[upper]
        * fraction
    )


def build_synthetic_instance(
    candidate_count,
    rule_density,
    rng,
):
    facts = [
        f"F{i:03d}"
        for i
        in range(
            1,
            candidate_count + 1,
        )
    ]

    weights = {
        fact_id: float(
            rng.randint(
                1,
                8,
            )
        )
        for fact_id
        in facts
    }

    target_rule_count = max(
        candidate_count,
        int(
            math.ceil(
                candidate_count
                * rule_density
            )
        ),
    )

    edge_set = set()

    # Base cycle ensures that every fact participates
    # and that the rule graph is connected.
    for index in range(
        candidate_count
    ):
        left = facts[
            index
        ]

        right = facts[
            (
                index + 1
            )
            % candidate_count
        ]

        edge = tuple(
            sorted(
                [
                    left,
                    right,
                ]
            )
        )

        edge_set.add(
            edge
        )

    # Additional overlapping 2-4 fact rules create
    # denser connected optimization cases.
    attempts = 0
    maximum_attempts = (
        target_rule_count
        * 100
    )

    while (
        len(edge_set)
        < target_rule_count
        and attempts
        < maximum_attempts
    ):
        attempts += 1

        maximum_edge_size = min(
            4,
            candidate_count,
        )

        edge_size = rng.randint(
            2,
            maximum_edge_size,
        )

        edge = tuple(
            sorted(
                rng.sample(
                    facts,
                    edge_size,
                )
            )
        )

        edge_set.add(
            edge
        )

    edges = [
        list(edge)
        for edge
        in sorted(
            edge_set
        )
    ]

    return {
        "facts": facts,
        "weights": weights,
        "edges": edges,
    }


def benchmark_solver(
    edges,
    weights,
    repeats,
):
    # Untimed warm-up.
    warm_removed, _ = (
        solve_minimum_hitting_set(
            edges=edges,
            weights=weights,
        )
    )

    if not safety_preserved(
        warm_removed,
        edges,
    ):
        raise RuntimeError(
            "Warm-up solution violated "
            "a security constraint."
        )

    elapsed_ms = []

    removed_counts = []

    optimal_costs = []

    for _ in range(
        repeats
    ):
        start = (
            time.perf_counter_ns()
        )

        (
            removed,
            optimal_cost,
        ) = (
            solve_minimum_hitting_set(
                edges=edges,
                weights=weights,
            )
        )

        end = (
            time.perf_counter_ns()
        )

        if not safety_preserved(
            removed,
            edges,
        ):
            raise RuntimeError(
                "Solver produced a solution "
                "that violated a rule."
            )

        elapsed_ms.append(
            (
                end - start
            )
            / 1_000_000.0
        )

        removed_counts.append(
            len(
                removed
            )
        )

        optimal_costs.append(
            float(
                optimal_cost
            )
        )

    return {
        "elapsed_ms": elapsed_ms,
        "removed_counts": (
            removed_counts
        ),
        "optimal_costs": (
            optimal_costs
        ),
    }


def run_synthetic_benchmark(
    sizes,
    instances,
    timing_repeats,
    rule_density,
    seed,
):
    rows = []

    for candidate_count in sizes:
        for instance_index in range(
            1,
            instances + 1,
        ):
            instance_seed = (
                seed
                + candidate_count
                * 100_000
                + instance_index
            )

            rng = random.Random(
                instance_seed
            )

            instance = (
                build_synthetic_instance(
                    candidate_count=(
                        candidate_count
                    ),
                    rule_density=(
                        rule_density
                    ),
                    rng=rng,
                )
            )

            result = benchmark_solver(
                edges=instance[
                    "edges"
                ],
                weights=instance[
                    "weights"
                ],
                repeats=(
                    timing_repeats
                ),
            )

            for repeat_index, (
                elapsed_ms,
                removed_count,
                optimal_cost,
            ) in enumerate(
                zip(
                    result[
                        "elapsed_ms"
                    ],
                    result[
                        "removed_counts"
                    ],
                    result[
                        "optimal_costs"
                    ],
                ),
                start=1,
            ):
                rows.append(
                    {
                        "candidate_fact_count": (
                            candidate_count
                        ),
                        "rule_count": len(
                            instance[
                                "edges"
                            ]
                        ),
                        "instance": (
                            instance_index
                        ),
                        "repeat": (
                            repeat_index
                        ),
                        "instance_seed": (
                            instance_seed
                        ),
                        "removed_fact_count": (
                            removed_count
                        ),
                        "optimal_cost": (
                            optimal_cost
                        ),
                        "elapsed_ms": (
                            elapsed_ms
                        ),
                        "safety_preserved": (
                            1
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def summarize_synthetic(
    trial_df,
):
    rows = []

    grouped = trial_df.groupby(
        "candidate_fact_count",
        sort=True,
    )

    for (
        candidate_count,
        group,
    ) in grouped:
        timings = (
            group[
                "elapsed_ms"
            ]
            .astype(float)
            .tolist()
        )

        rows.append(
            {
                "candidate_fact_count": (
                    int(
                        candidate_count
                    )
                ),
                "rule_count_mean": (
                    float(
                        group[
                            "rule_count"
                        ].mean()
                    )
                ),
                "instances": int(
                    group[
                        "instance"
                    ].nunique()
                ),
                "timed_runs": int(
                    len(
                        group
                    )
                ),
                "median_ms": (
                    statistics.median(
                        timings
                    )
                ),
                "mean_ms": (
                    statistics.mean(
                        timings
                    )
                ),
                "p95_ms": (
                    percentile(
                        timings,
                        0.95,
                    )
                ),
                "p99_ms": (
                    percentile(
                        timings,
                        0.99,
                    )
                ),
                "max_ms": (
                    max(
                        timings
                    )
                ),
                "mean_removed_fact_count": (
                    float(
                        group[
                            "removed_fact_count"
                        ].mean()
                    )
                ),
                "safety_preserved_rate": (
                    float(
                        group[
                            "safety_preserved"
                        ].mean()
                    )
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary[
        "median_growth_vs_previous"
    ] = float(
        "nan"
    )

    for index in range(
        1,
        len(
            summary
        ),
    ):
        previous = float(
            summary.loc[
                index - 1,
                "median_ms",
            ]
        )

        current = float(
            summary.loc[
                index,
                "median_ms",
            ]
        )

        if previous > 0:
            summary.loc[
                index,
                "median_growth_vs_previous",
            ] = (
                current
                / previous
            )

    return summary


def run_actual_component_benchmark(
    rule_file,
    utility_file,
    repeats,
):
    rules = load_rules(
        rule_file
    )

    utility_map = (
        load_utility_map(
            utility_file
        )
    )

    validate_utility_coverage(
        rules=rules,
        utility_map=utility_map,
    )

    components = (
        build_rule_components(
            rules
        )
    )

    rows = []

    for component_index, component in enumerate(
        components,
        start=1,
    ):
        edges = [
            rule[
                "required_facts"
            ]
            for rule
            in component
        ]

        facts = sorted(
            {
                fact_id
                for edge
                in edges
                for fact_id
                in edge
            }
        )

        result = benchmark_solver(
            edges=edges,
            weights=utility_map,
            repeats=repeats,
        )

        timings = result[
            "elapsed_ms"
        ]

        rows.append(
            {
                "component_id": (
                    f"COMP-{component_index:03d}"
                ),
                "candidate_fact_count": (
                    len(
                        facts
                    )
                ),
                "rule_count": (
                    len(
                        edges
                    )
                ),
                "timed_runs": (
                    repeats
                ),
                "median_ms": (
                    statistics.median(
                        timings
                    )
                ),
                "mean_ms": (
                    statistics.mean(
                        timings
                    )
                ),
                "p95_ms": (
                    percentile(
                        timings,
                        0.95,
                    )
                ),
                "p99_ms": (
                    percentile(
                        timings,
                        0.99,
                    )
                ),
                "max_ms": (
                    max(
                        timings
                    )
                ),
                "removed_fact_count": (
                    result[
                        "removed_counts"
                    ][0]
                ),
                "optimal_cost": (
                    result[
                        "optimal_costs"
                    ][0]
                ),
                "safety_preserved": (
                    1
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def summarize_actual_components(
    actual_df,
):
    timings = (
        actual_df[
            "median_ms"
        ]
        .astype(float)
        .tolist()
    )

    if not timings:
        raise ValueError(
            "No actual rule components "
            "were benchmarked."
        )

    return pd.DataFrame(
        [
            {
                "component_count": (
                    len(
                        actual_df
                    )
                ),
                "min_candidate_fact_count": (
                    int(
                        actual_df[
                            "candidate_fact_count"
                        ].min()
                    )
                ),
                "max_candidate_fact_count": (
                    int(
                        actual_df[
                            "candidate_fact_count"
                        ].max()
                    )
                ),
                "min_rule_count": (
                    int(
                        actual_df[
                            "rule_count"
                        ].min()
                    )
                ),
                "max_rule_count": (
                    int(
                        actual_df[
                            "rule_count"
                        ].max()
                    )
                ),
                "median_component_median_ms": (
                    statistics.median(
                        timings
                    )
                ),
                "p95_component_median_ms": (
                    percentile(
                        timings,
                        0.95,
                    )
                ),
                "max_component_median_ms": (
                    max(
                        timings
                    )
                ),
                "safety_preserved_rate": (
                    float(
                        actual_df[
                            "safety_preserved"
                        ].mean()
                    )
                ),
            }
        ]
    )


def build_paper_table(
    synthetic_summary,
):
    rows = []

    for _, row in (
        synthetic_summary.iterrows()
    ):
        rows.append(
            {
                "Candidate facts": (
                    int(
                        row[
                            "candidate_fact_count"
                        ]
                    )
                ),
                "Mean rules": (
                    f"{float(row['rule_count_mean']):.1f}"
                ),
                "Timed runs": (
                    int(
                        row[
                            "timed_runs"
                        ]
                    )
                ),
                "Median (ms)": (
                    f"{float(row['median_ms']):.4f}"
                ),
                "P95 (ms)": (
                    f"{float(row['p95_ms']):.4f}"
                ),
                "P99 (ms)": (
                    f"{float(row['p99_ms']):.4f}"
                ),
                "Max (ms)": (
                    f"{float(row['max_ms']):.4f}"
                ),
                "Safety preserved": (
                    f"{float(row['safety_preserved_rate']) * 100:.2f}%"
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    args = parse_args()

    if any(
        size < 2
        for size
        in args.sizes
    ):
        raise ValueError(
            "All candidate-fact counts "
            "must be at least 2."
        )

    if args.instances <= 0:
        raise ValueError(
            "--instances must be positive."
        )

    if args.timing_repeats <= 0:
        raise ValueError(
            "--timing-repeats must be positive."
        )

    if args.actual_repeats <= 0:
        raise ValueError(
            "--actual-repeats must be positive."
        )

    if args.rule_density < 1.0:
        raise ValueError(
            "--rule-density should be "
            "at least 1.0."
        )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rule_file = (
        args.rule_file.resolve()
    )

    utility_file = (
        args.utility_file.resolve()
    )

    synthetic_trials = (
        run_synthetic_benchmark(
            sizes=args.sizes,
            instances=args.instances,
            timing_repeats=(
                args.timing_repeats
            ),
            rule_density=(
                args.rule_density
            ),
            seed=args.seed,
        )
    )

    synthetic_summary = (
        summarize_synthetic(
            synthetic_trials
        )
    )

    actual_components = (
        run_actual_component_benchmark(
            rule_file=rule_file,
            utility_file=utility_file,
            repeats=(
                args.actual_repeats
            ),
        )
    )

    actual_summary = (
        summarize_actual_components(
            actual_components
        )
    )

    paper_table = (
        build_paper_table(
            synthetic_summary
        )
    )

    synthetic_trial_path = (
        output_dir
        / "synthetic_trials.csv"
    )

    synthetic_summary_path = (
        output_dir
        / "synthetic_summary.csv"
    )

    actual_component_path = (
        output_dir
        / "actual_rule_components.csv"
    )

    actual_summary_path = (
        output_dir
        / "actual_rule_summary.csv"
    )

    paper_table_path = (
        output_dir
        / "paper_scalability_table.csv"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    synthetic_trials.to_csv(
        synthetic_trial_path,
        index=False,
        encoding="utf-8-sig",
    )

    synthetic_summary.to_csv(
        synthetic_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    actual_components.to_csv(
        actual_component_path,
        index=False,
        encoding="utf-8-sig",
    )

    actual_summary.to_csv(
        actual_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    paper_table.to_csv(
        paper_table_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "analysis": (
            "controller optimization kernel "
            "scalability benchmark"
        ),
        "solver": (
            "src.utility_sensitivity_analysis."
            "solve_minimum_hitting_set"
        ),
        "scope": (
            "Exact minimum-weight hitting-set "
            "optimization only; retrieval, LLM generation, "
            "output guard, and network latency are excluded."
        ),
        "synthetic_sizes": (
            args.sizes
        ),
        "instances_per_size": (
            args.instances
        ),
        "timing_repeats_per_instance": (
            args.timing_repeats
        ),
        "actual_component_repeats": (
            args.actual_repeats
        ),
        "rule_density": (
            args.rule_density
        ),
        "seed": (
            args.seed
        ),
        "synthetic_rule_structure": (
            "Connected cycle plus randomly sampled "
            "overlapping 2-4 fact rules."
        ),
        "timing_method": (
            "Python time.perf_counter_ns; "
            "case generation and warm-up are excluded."
        ),
        "interpretation_notes": [
            (
                "This benchmark measures only the exact "
                "minimum-loss optimization kernel."
            ),
            (
                "Runtime depends on both candidate-fact "
                "count and forbidden-rule structure."
            ),
            (
                "The synthetic benchmark is a controlled "
                "stress test and is not a deployment "
                "latency measurement."
            ),
            (
                "The exact solver has exponential "
                "worst-case complexity; the empirical "
                "benchmark does not change that bound."
            ),
            (
                "Actual-rule-component timings use the "
                "same 60-rule full experiment dataset."
            ),
        ],
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

    print()
    print(
        "=" * 72
    )
    print(
        "CONTROLLER SCALABILITY ANALYSIS COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        "Synthetic sizes:",
        args.sizes,
    )

    print(
        "Instances per size:",
        args.instances,
    )

    print(
        "Timing repeats per instance:",
        args.timing_repeats,
    )

    print(
        "Actual rule components:",
        len(
            actual_components
        ),
    )

    print()
    print(
        "Generated:"
    )

    for path in [
        synthetic_trial_path,
        synthetic_summary_path,
        actual_component_path,
        actual_summary_path,
        paper_table_path,
        metadata_path,
    ]:
        print(
            "-",
            path,
        )

    print()
    print(
        "Synthetic scalability summary:"
    )

    print(
        synthetic_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Actual rule-set summary:"
    )

    print(
        actual_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()