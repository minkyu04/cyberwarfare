import argparse
import json
import math
import random
from itertools import combinations
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_UTILITY_FILE = (
    BASE_DIR
    / "data"
    / "full_experiment"
    / "fact_utility.json"
)

DEFAULT_RULE_FILE = (
    BASE_DIR
    / "data"
    / "full_experiment"
    / "inference_rules.json"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "utility_sensitivity"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Utility-weight sensitivity analysis for the "
            "weighted minimum-loss disclosure controller."
        )
    )

    parser.add_argument(
        "--utility-file",
        type=Path,
        default=DEFAULT_UTILITY_FILE,
    )

    parser.add_argument(
        "--rule-file",
        type=Path,
        default=DEFAULT_RULE_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[
            0.10,
            0.20,
            0.30,
            0.50,
        ],
        help=(
            "Multiplicative perturbation half-width. "
            "0.10 means +/-10 percent."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
    ) as f:
        return json.load(f)


def load_utility_map(path):
    raw = load_json(path)

    if isinstance(raw, list):
        utility_map = {}

        for item in raw:
            fact_id = item["fact_id"]
            business_value = float(
                item["business_value"]
            )

            utility_map[fact_id] = (
                business_value
            )

        return utility_map

    if isinstance(raw, dict):
        utility_map = {}

        for fact_id, value in raw.items():
            if isinstance(value, dict):
                value = value[
                    "business_value"
                ]

            utility_map[fact_id] = float(
                value
            )

        return utility_map

    raise TypeError(
        "fact_utility.json must be "
        "a list or dictionary."
    )


def load_rules(path):
    raw_rules = load_json(path)

    if not isinstance(raw_rules, list):
        raise TypeError(
            "inference_rules.json must be a list."
        )

    rules = []

    for index, rule in enumerate(
        raw_rules,
        start=1,
    ):
        required_facts = list(
            dict.fromkeys(
                rule.get(
                    "required_facts",
                    [],
                )
            )
        )

        if len(required_facts) < 2:
            continue

        rules.append(
            {
                "rule_id": rule.get(
                    "rule_id",
                    f"RULE-{index:03d}",
                ),
                "required_facts": (
                    required_facts
                ),
                "description": rule.get(
                    "description",
                    "",
                ),
            }
        )

    if not rules:
        raise ValueError(
            "No inference rules containing "
            "two or more required facts."
        )

    return rules


def validate_utility_coverage(
    rules,
    utility_map,
):
    required_fact_ids = {
        fact_id
        for rule in rules
        for fact_id
        in rule["required_facts"]
    }

    missing = sorted(
        required_fact_ids
        - set(utility_map)
    )

    if missing:
        raise ValueError(
            "Utility values are missing for: "
            + ", ".join(
                missing[:20]
            )
        )


def build_rule_components(rules):
    fact_to_rule_indices = {}

    for rule_index, rule in enumerate(
        rules
    ):
        for fact_id in rule[
            "required_facts"
        ]:
            fact_to_rule_indices.setdefault(
                fact_id,
                set(),
            ).add(
                rule_index
            )

    visited = set()
    components = []

    for start_index in range(
        len(rules)
    ):
        if start_index in visited:
            continue

        stack = [
            start_index
        ]

        component_indices = set()

        while stack:
            current_index = (
                stack.pop()
            )

            if current_index in visited:
                continue

            visited.add(
                current_index
            )

            component_indices.add(
                current_index
            )

            current_rule = rules[
                current_index
            ]

            for fact_id in current_rule[
                "required_facts"
            ]:
                neighbors = (
                    fact_to_rule_indices[
                        fact_id
                    ]
                )

                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(
                            neighbor
                        )

        components.append(
            [
                rules[index]
                for index
                in sorted(
                    component_indices
                )
            ]
        )

    return components


def cost_of(
    selected_facts,
    weights,
):
    return sum(
        weights[fact_id]
        for fact_id
        in selected_facts
    )


def solution_key(
    selected_facts,
    weights,
):
    ordered = tuple(
        sorted(
            selected_facts
        )
    )

    total_cost = cost_of(
        ordered,
        weights,
    )

    return (
        total_cost,
        len(ordered),
        ordered,
    )


def solve_minimum_hitting_set(
    edges,
    weights,
):
    edges = [
        frozenset(edge)
        for edge in edges
        if edge
    ]

    if not edges:
        return set(), 0.0

    best_selected = None
    best_key = None

    memo = {}

    def recurse(
        selected,
        uncovered,
        running_cost,
    ):
        nonlocal best_selected
        nonlocal best_key

        if best_key is not None:
            if (
                running_cost
                > best_key[0]
                + 1e-12
            ):
                return

        if not uncovered:
            current_key = solution_key(
                selected,
                weights,
            )

            if (
                best_key is None
                or current_key
                < best_key
            ):
                best_key = (
                    current_key
                )

                best_selected = set(
                    selected
                )

            return

        signature = tuple(
            sorted(
                tuple(
                    sorted(edge)
                )
                for edge
                in uncovered
            )
        )

        previous_cost = memo.get(
            signature
        )

        if (
            previous_cost is not None
            and running_cost
            > previous_cost
            + 1e-12
        ):
            return

        if (
            previous_cost is None
            or running_cost
            < previous_cost
        ):
            memo[
                signature
            ] = running_cost

        branch_edge = min(
            uncovered,
            key=lambda edge: (
                len(edge),
                sum(
                    weights[fact_id]
                    for fact_id
                    in edge
                ),
                tuple(
                    sorted(edge)
                ),
            ),
        )

        branch_facts = sorted(
            branch_edge,
            key=lambda fact_id: (
                weights[fact_id],
                fact_id,
            ),
        )

        for fact_id in branch_facts:
            new_selected = (
                selected
                | {
                    fact_id
                }
            )

            new_cost = (
                running_cost
                + weights[
                    fact_id
                ]
            )

            if (
                best_key is not None
                and new_cost
                > best_key[0]
                + 1e-12
            ):
                continue

            new_uncovered = [
                edge
                for edge
                in uncovered
                if fact_id
                not in edge
            ]

            recurse(
                new_selected,
                new_uncovered,
                new_cost,
            )

    recurse(
        selected=set(),
        uncovered=edges,
        running_cost=0.0,
    )

    if best_selected is None:
        raise RuntimeError(
            "Minimum hitting-set "
            "optimization failed."
        )

    return (
        best_selected,
        float(
            best_key[0]
        ),
    )


def perturb_weights(
    base_weights,
    facts,
    level,
    rng,
):
    perturbed = dict(
        base_weights
    )

    for fact_id in facts:
        multiplier = (
            1.0
            + rng.uniform(
                -level,
                level,
            )
        )

        perturbed[
            fact_id
        ] = max(
            1e-9,
            base_weights[
                fact_id
            ]
            * multiplier,
        )

    return perturbed


def safety_preserved(
    selected_facts,
    edges,
):
    selected = set(
        selected_facts
    )

    return all(
        bool(
            selected
            & set(edge)
        )
        for edge in edges
    )


def compare_pairwise_ranking(
    facts,
    base_weights,
    perturbed_weights,
):
    epsilon = 1e-12

    tied_pair_count = 0
    tie_resolved_count = 0

    strictly_ordered_pair_count = 0
    strict_rank_reversal_count = 0

    for left, right in combinations(
        sorted(facts),
        2,
    ):
        base_difference = (
            base_weights[left]
            - base_weights[right]
        )

        perturbed_difference = (
            perturbed_weights[left]
            - perturbed_weights[right]
        )

        base_is_tie = (
            abs(
                base_difference
            )
            <= epsilon
        )

        perturbed_is_tie = (
            abs(
                perturbed_difference
            )
            <= epsilon
        )

        if base_is_tie:
            tied_pair_count += 1

            if not perturbed_is_tie:
                tie_resolved_count += 1

            continue

        strictly_ordered_pair_count += 1

        if (
            not perturbed_is_tie
            and (
                base_difference
                * perturbed_difference
                < 0
            )
        ):
            strict_rank_reversal_count += 1

    tie_resolution_event = int(
        tie_resolved_count > 0
    )

    strict_rank_reversal_event = int(
        strict_rank_reversal_count > 0
    )

    any_rank_change_event = int(
        (
            tie_resolved_count > 0
        )
        or (
            strict_rank_reversal_count > 0
        )
    )

    tie_resolution_pair_rate = (
        tie_resolved_count
        / tied_pair_count
        if tied_pair_count > 0
        else 0.0
    )

    strict_rank_reversal_pair_rate = (
        strict_rank_reversal_count
        / strictly_ordered_pair_count
        if strictly_ordered_pair_count > 0
        else 0.0
    )

    return {
        "pairwise_rank_changed": (
            any_rank_change_event
        ),
        "tie_resolution_event": (
            tie_resolution_event
        ),
        "strict_rank_reversal_event": (
            strict_rank_reversal_event
        ),
        "tied_pair_count": (
            tied_pair_count
        ),
        "tie_resolved_count": (
            tie_resolved_count
        ),
        "tie_resolution_pair_rate": (
            tie_resolution_pair_rate
        ),
        "strictly_ordered_pair_count": (
            strictly_ordered_pair_count
        ),
        "strict_rank_reversal_count": (
            strict_rank_reversal_count
        ),
        "strict_rank_reversal_pair_rate": (
            strict_rank_reversal_pair_rate
        ),
    }


def wilson_interval(
    events,
    n,
    z=1.96,
):
    if n == 0:
        return (
            float("nan"),
            float("nan"),
        )

    proportion = (
        events
        / n
    )

    denominator = (
        1.0
        + z * z / n
    )

    center = (
        proportion
        + z * z
        / (
            2.0
            * n
        )
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion
            * (
                1.0
                - proportion
            )
            / n
            + z * z
            / (
                4.0
                * n
                * n
            )
        )
        / denominator
    )

    return (
        max(
            0.0,
            center - margin,
        ),
        min(
            1.0,
            center + margin,
        ),
    )


def mean_ci95(values):
    series = pd.Series(
        values,
        dtype="float64",
    ).dropna()

    n = len(
        series
    )

    if n == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    mean = float(
        series.mean()
    )

    if n == 1:
        return (
            mean,
            mean,
            mean,
        )

    standard_error = (
        float(
            series.std(
                ddof=1
            )
        )
        / math.sqrt(n)
    )

    margin = (
        1.96
        * standard_error
    )

    return (
        mean,
        mean - margin,
        mean + margin,
    )


def build_cases(
    components,
    utility_map,
):
    cases = []
    rows = []

    for case_index, rules in enumerate(
        components,
        start=1,
    ):
        case_id = (
            f"SENS-{case_index:03d}"
        )

        edges = [
            rule[
                "required_facts"
            ]
            for rule
            in rules
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

        (
            baseline_removed,
            baseline_cost,
        ) = solve_minimum_hitting_set(
            edges=edges,
            weights=utility_map,
        )

        total_utility = cost_of(
            facts,
            utility_map,
        )

        baseline_pairwise = (
            compare_pairwise_ranking(
                facts=facts,
                base_weights=utility_map,
                perturbed_weights=utility_map,
            )
        )

        cases.append(
            {
                "case_id": (
                    case_id
                ),
                "rules": (
                    rules
                ),
                "edges": (
                    edges
                ),
                "facts": (
                    facts
                ),
                "baseline_removed": (
                    baseline_removed
                ),
                "baseline_cost": (
                    baseline_cost
                ),
                "total_utility": (
                    total_utility
                ),
            }
        )

        rows.append(
            {
                "case_id": (
                    case_id
                ),
                "rule_count": len(
                    rules
                ),
                "fact_count": len(
                    facts
                ),
                "rule_ids": "|".join(
                    rule[
                        "rule_id"
                    ]
                    for rule
                    in rules
                ),
                "fact_ids": "|".join(
                    facts
                ),
                "baseline_removed_facts": (
                    "|".join(
                        sorted(
                            baseline_removed
                        )
                    )
                ),
                "baseline_removed_utility": (
                    baseline_cost
                ),
                "baseline_total_utility": (
                    total_utility
                ),
                "baseline_retention": (
                    1.0
                    - baseline_cost
                    / total_utility
                    if total_utility > 0
                    else float(
                        "nan"
                    )
                ),
                "baseline_tied_pair_count": (
                    baseline_pairwise[
                        "tied_pair_count"
                    ]
                ),
                "baseline_strictly_ordered_pair_count": (
                    baseline_pairwise[
                        "strictly_ordered_pair_count"
                    ]
                ),
            }
        )

    return (
        cases,
        rows,
    )


def run_trials(
    cases,
    utility_map,
    levels,
    repeats,
    seed,
):
    rng = random.Random(
        seed
    )

    rows = []

    for level in levels:
        if level < 0:
            raise ValueError(
                "Perturbation levels "
                "must be non-negative."
            )

        for repeat_index in range(
            1,
            repeats + 1,
        ):
            for case in cases:
                perturbed_weights = (
                    perturb_weights(
                        base_weights=(
                            utility_map
                        ),
                        facts=case[
                            "facts"
                        ],
                        level=level,
                        rng=rng,
                    )
                )

                (
                    selected,
                    perturbed_optimal_cost,
                ) = (
                    solve_minimum_hitting_set(
                        edges=case[
                            "edges"
                        ],
                        weights=(
                            perturbed_weights
                        ),
                    )
                )

                selected_cost_original = (
                    cost_of(
                        selected,
                        utility_map,
                    )
                )

                baseline_choice_cost_perturbed = (
                    cost_of(
                        case[
                            "baseline_removed"
                        ],
                        perturbed_weights,
                    )
                )

                baseline_regret = max(
                    0.0,
                    selected_cost_original
                    - case[
                        "baseline_cost"
                    ],
                )

                total_utility = case[
                    "total_utility"
                ]

                exact_agreement = int(
                    selected
                    == case[
                        "baseline_removed"
                    ]
                )

                optimal_equivalent = int(
                    abs(
                        selected_cost_original
                        - case[
                            "baseline_cost"
                        ]
                    )
                    <= 1e-12
                )

                changed_but_equivalent = int(
                    (
                        exact_agreement
                        == 0
                    )
                    and (
                        optimal_equivalent
                        == 1
                    )
                )

                rank_result = (
                    compare_pairwise_ranking(
                        facts=case[
                            "facts"
                        ],
                        base_weights=(
                            utility_map
                        ),
                        perturbed_weights=(
                            perturbed_weights
                        ),
                    )
                )

                row = {
                    "perturbation_level": (
                        level
                    ),
                    "repeat": (
                        repeat_index
                    ),
                    "case_id": (
                        case[
                            "case_id"
                        ]
                    ),
                    "safety_preserved": int(
                        safety_preserved(
                            selected,
                            case[
                                "edges"
                            ],
                        )
                    ),
                    "exact_selection_agreement": (
                        exact_agreement
                    ),
                    "baseline_optimal_equivalent": (
                        optimal_equivalent
                    ),
                    "selection_changed_but_equivalent": (
                        changed_but_equivalent
                    ),
                    "baseline_removed_facts": (
                        "|".join(
                            sorted(
                                case[
                                    "baseline_removed"
                                ]
                            )
                        )
                    ),
                    "selected_removed_facts": (
                        "|".join(
                            sorted(
                                selected
                            )
                        )
                    ),
                    "baseline_optimal_cost": (
                        case[
                            "baseline_cost"
                        ]
                    ),
                    "selected_cost_under_baseline": (
                        selected_cost_original
                    ),
                    "baseline_regret_abs": (
                        baseline_regret
                    ),
                    "baseline_regret_relative_total": (
                        baseline_regret
                        / total_utility
                        if total_utility > 0
                        else float(
                            "nan"
                        )
                    ),
                    "original_utility_retention": (
                        1.0
                        - selected_cost_original
                        / total_utility
                        if total_utility > 0
                        else float(
                            "nan"
                        )
                    ),
                    "perturbed_optimal_cost": (
                        perturbed_optimal_cost
                    ),
                    "baseline_choice_cost_under_perturbed": (
                        baseline_choice_cost_perturbed
                    ),
                    "perturbed_decision_gain": max(
                        0.0,
                        baseline_choice_cost_perturbed
                        - perturbed_optimal_cost,
                    ),
                }

                row.update(
                    rank_result
                )

                rows.append(
                    row
                )

    return rows


def summarize_trials(
    trial_df,
):
    summary_rows = []

    rate_columns = [
        "safety_preserved",
        "exact_selection_agreement",
        "baseline_optimal_equivalent",
        "selection_changed_but_equivalent",
        "pairwise_rank_changed",
        "tie_resolution_event",
        "strict_rank_reversal_event",
    ]

    continuous_columns = [
        "tie_resolution_pair_rate",
        "strict_rank_reversal_pair_rate",
        "baseline_regret_abs",
        "baseline_regret_relative_total",
        "original_utility_retention",
        "perturbed_decision_gain",
    ]

    grouped = trial_df.groupby(
        "perturbation_level",
        sort=True,
    )

    for level, group in grouped:
        row = {
            "perturbation_level": (
                level
            ),
            "n_trials": len(
                group
            ),
            "n_cases": group[
                "case_id"
            ].nunique(),
            "repeats": group[
                "repeat"
            ].nunique(),
        }

        for column in rate_columns:
            events = int(
                group[
                    column
                ].sum()
            )

            n = int(
                group[
                    column
                ].count()
            )

            (
                ci_low,
                ci_high,
            ) = wilson_interval(
                events,
                n,
            )

            row[
                f"{column}_rate"
            ] = (
                events / n
                if n > 0
                else float(
                    "nan"
                )
            )

            row[
                f"{column}_ci95_low"
            ] = ci_low

            row[
                f"{column}_ci95_high"
            ] = ci_high

        for column in continuous_columns:
            (
                mean,
                ci_low,
                ci_high,
            ) = mean_ci95(
                group[
                    column
                ]
            )

            row[
                f"mean_{column}"
            ] = mean

            row[
                f"{column}_ci95_low"
            ] = ci_low

            row[
                f"{column}_ci95_high"
            ] = ci_high

            row[
                f"median_{column}"
            ] = float(
                group[
                    column
                ].median()
            )

            row[
                f"p95_{column}"
            ] = float(
                group[
                    column
                ].quantile(
                    0.95
                )
            )

        summary_rows.append(
            row
        )

    return pd.DataFrame(
        summary_rows
    )


def build_compact_summary(
    summary_df,
):
    selected_columns = [
        "perturbation_level",
        "n_trials",
        "safety_preserved_rate",
        "exact_selection_agreement_rate",
        "baseline_optimal_equivalent_rate",
        "selection_changed_but_equivalent_rate",
        "tie_resolution_event_rate",
        "strict_rank_reversal_event_rate",
        "mean_tie_resolution_pair_rate",
        "mean_strict_rank_reversal_pair_rate",
        "mean_baseline_regret_relative_total",
        "mean_original_utility_retention",
    ]

    return summary_df[
        selected_columns
    ].copy()


def main():
    args = parse_args()

    if args.repeats <= 0:
        raise ValueError(
            "--repeats must be "
            "greater than zero."
        )

    utility_file = (
        args.utility_file.resolve()
    )

    rule_file = (
        args.rule_file.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    utility_map = load_utility_map(
        utility_file
    )

    rules = load_rules(
        rule_file
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

    (
        cases,
        case_rows,
    ) = build_cases(
        components=components,
        utility_map=utility_map,
    )

    trial_rows = run_trials(
        cases=cases,
        utility_map=utility_map,
        levels=args.levels,
        repeats=args.repeats,
        seed=args.seed,
    )

    case_df = pd.DataFrame(
        case_rows
    )

    trial_df = pd.DataFrame(
        trial_rows
    )

    summary_df = summarize_trials(
        trial_df
    )

    compact_summary_df = (
        build_compact_summary(
            summary_df
        )
    )

    case_path = (
        output_dir
        / "case_definition.csv"
    )

    trial_path = (
        output_dir
        / "trial_results.csv"
    )

    summary_path = (
        output_dir
        / "summary.csv"
    )

    compact_summary_path = (
        output_dir
        / "compact_summary.csv"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    case_df.to_csv(
        case_path,
        index=False,
        encoding="utf-8-sig",
    )

    trial_df.to_csv(
        trial_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    compact_summary_df.to_csv(
        compact_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "analysis": (
            "controller-level utility sensitivity"
        ),
        "utility_file": str(
            utility_file
        ),
        "rule_file": str(
            rule_file
        ),
        "rule_count": len(
            rules
        ),
        "case_count": len(
            cases
        ),
        "utility_fact_count": len(
            utility_map
        ),
        "perturbation_levels": (
            args.levels
        ),
        "repeats": (
            args.repeats
        ),
        "seed": (
            args.seed
        ),
        "perturbation_method": (
            "Independent multiplicative uniform perturbation: "
            "w' = w * (1 + U[-level, level])."
        ),
        "optimization": (
            "Exact minimum-weight hitting set for each "
            "connected component of overlapping inference rules."
        ),
        "tie_break": (
            "Minimum cost, then minimum number of removals, "
            "then lexicographic fact ID order."
        ),
        "rank_metrics": {
            "tie_resolution_event": (
                "At least one pair with equal baseline utility "
                "becomes non-equal after perturbation."
            ),
            "strict_rank_reversal_event": (
                "At least one pair with unequal baseline utility "
                "reverses its utility ordering after perturbation."
            ),
            "tie_resolution_pair_rate": (
                "Fraction of baseline-tied fact pairs whose ties "
                "are resolved by perturbation."
            ),
            "strict_rank_reversal_pair_rate": (
                "Fraction of strictly ordered baseline fact pairs "
                "whose ordering is reversed by perturbation."
            ),
        },
        "interpretation_notes": [
            (
                "This is a controller-level robustness analysis "
                "and does not rerun LLM generation."
            ),
            (
                "Security constraints remain hard constraints. "
                "Utility perturbation changes only which facts "
                "are selected for removal."
            ),
            (
                "baseline_optimal_equivalent counts alternative "
                "removal sets with the same original minimum cost "
                "as equivalent."
            ),
            (
                "selection_changed_but_equivalent identifies "
                "runs where the selected fact IDs changed while "
                "the original minimum-loss objective value did not."
            ),
            (
                "Reported 95 percent intervals summarize Monte "
                "Carlo perturbation-case observations and should "
                "not be interpreted as deployment-population "
                "confidence intervals."
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
        "UTILITY SENSITIVITY ANALYSIS COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        "Rules:",
        len(rules),
    )

    print(
        "Connected rule cases:",
        len(cases),
    )

    print(
        "Utility facts:",
        len(utility_map),
    )

    print(
        "Perturbation levels:",
        args.levels,
    )

    print(
        "Repeats:",
        args.repeats,
    )

    print(
        "Seed:",
        args.seed,
    )

    print()
    print(
        "Generated:"
    )

    print(
        "-",
        case_path,
    )

    print(
        "-",
        trial_path,
    )

    print(
        "-",
        summary_path,
    )

    print(
        "-",
        compact_summary_path,
    )

    print(
        "-",
        metadata_path,
    )

    print()
    print(
        "Compact summary:"
    )

    print(
        compact_summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()