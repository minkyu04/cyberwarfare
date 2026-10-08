import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULT_DIR / "final_paper_tables"


PATHS = {
    "qwen05_test_analysis": (
        RESULT_DIR
        / "analysis_qwen25_05b_final_test_v3"
    ),
    "qwen05_dev_analysis": (
        RESULT_DIR
        / "analysis_qwen25_05b_full_dev_v3"
    ),
    "qwen3_test_analysis": (
        RESULT_DIR
        / "analysis_qwen25_3b_crossmodel_at7_at8_test"
    ),
    "phi35_test_analysis": (
        RESULT_DIR
        / "analysis_phi35_crossmodel_at7_at8_test"
    ),
    "qwen05_test_summary": (
        RESULT_DIR
        / "qwen25_05b_final_test_v3_summary.csv"
    ),
    "qwen05_dev_summary": (
        RESULT_DIR
        / "qwen25_05b_full_dev_v3_summary.csv"
    ),
    "ablation_no_history": (
        RESULT_DIR
        / "f_no_history_full_dev_summary.csv"
    ),
    "ablation_no_weight": (
        RESULT_DIR
        / "f_no_weight_full_dev_summary.csv"
    ),
    "ablation_all_block": (
        RESULT_DIR
        / "f_all_block_full_dev_summary.csv"
    ),
    "utility_sensitivity": (
        RESULT_DIR
        / "utility_sensitivity"
        / "compact_summary.csv"
    ),
}


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def read_csv(path):
    require_file(path)

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
    )


def read_analysis_file(
    analysis_dir,
    filename,
):
    return read_csv(
        analysis_dir
        / filename
    )


def as_percent(value):
    if pd.isna(value):
        return ""

    return (
        f"{float(value) * 100:.2f}%"
    )


def as_decimal(
    value,
    digits=4,
):
    if pd.isna(value):
        return ""

    return (
        f"{float(value):.{digits}f}"
    )


def format_binary_row(row):
    if row is None:
        return ""

    n = int(
        row["n"]
    )

    events = int(
        row["events"]
    )

    rate = (
        float(
            row["rate"]
        )
        * 100
    )

    low = (
        float(
            row["ci95_low"]
        )
        * 100
    )

    high = (
        float(
            row["ci95_high"]
        )
        * 100
    )

    return (
        f"{events}/{n} "
        f"({rate:.2f}%; "
        f"95% CI {low:.2f}–{high:.2f})"
    )


def format_continuous_row(
    row,
    percent=True,
):
    if row is None:
        return ""

    mean = float(
        row["mean"]
    )

    low = float(
        row["ci95_low"]
    )

    high = float(
        row["ci95_high"]
    )

    if percent:
        return (
            f"{mean * 100:.2f}% "
            f"(95% CI "
            f"{low * 100:.2f}–"
            f"{high * 100:.2f})"
        )

    return (
        f"{mean:.4f} "
        f"(95% CI "
        f"{low:.4f}–"
        f"{high:.4f})"
    )


def find_binary(
    df,
    condition,
    metric,
    scope=None,
):
    subset = df[
        (
            df["condition"]
            == condition
        )
        & (
            df["metric"]
            == metric
        )
    ]

    if scope is not None:
        subset = subset[
            subset[
                "analysis_scope"
            ]
            == scope
        ]

    if subset.empty:
        return None

    return subset.iloc[0]


def find_continuous(
    df,
    condition,
    metric,
    scope=None,
):
    subset = df[
        (
            df["condition"]
            == condition
        )
        & (
            df["metric"]
            == metric
        )
    ]

    if scope is not None:
        subset = subset[
            subset[
                "analysis_scope"
            ]
            == scope
        ]

    if subset.empty:
        return None

    return subset.iloc[0]


def build_table_1_overall_af():
    analysis_dir = PATHS[
        "qwen05_test_analysis"
    ]

    binary = read_analysis_file(
        analysis_dir,
        "overall_binary_summary.csv",
    )

    continuous = read_analysis_file(
        analysis_dir,
        "overall_continuous_summary.csv",
    )

    rows = []

    for condition in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]:
        evidence = find_binary(
            binary,
            condition,
            (
                "unauthorized_evidence_"
                "emission_event"
            ),
        )

        retrieval = find_binary(
            binary,
            condition,
            (
                "unauthorized_retrieval_"
                "exposure_event"
            ),
        )

        forbidden = find_binary(
            binary,
            condition,
            "forbidden_disclosure_event",
        )

        overblock = find_binary(
            binary,
            condition,
            "overblocking_event",
        )

        recall = find_continuous(
            continuous,
            condition,
            "authorized_fact_recall",
        )

        utility = find_continuous(
            continuous,
            condition,
            (
                "authorized_utility_"
                "retention_rate"
            ),
        )

        internal_utility = (
            find_continuous(
                continuous,
                condition,
                (
                    "internal_utility_"
                    "retention_rate"
                ),
            )
        )

        rows.append(
            {
                "Condition": (
                    condition
                ),
                "Unauthorized evidence emission": (
                    format_binary_row(
                        evidence
                    )
                ),
                "Unauthorized retrieval exposure": (
                    format_binary_row(
                        retrieval
                    )
                ),
                "Forbidden disclosure": (
                    format_binary_row(
                        forbidden
                    )
                ),
                "Overblocking": (
                    format_binary_row(
                        overblock
                    )
                ),
                "Authorized fact recall": (
                    format_continuous_row(
                        recall
                    )
                ),
                "Authorized utility retention": (
                    format_continuous_row(
                        utility
                    )
                ),
                "Internal weighted utility retention": (
                    format_continuous_row(
                        internal_utility
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_2_at7_at8():
    analysis_dir = PATHS[
        "qwen05_test_analysis"
    ]

    binary = read_analysis_file(
        analysis_dir,
        "scope_binary_summary.csv",
    )

    rows = []

    metrics = [
        (
            "AT7 cross-document",
            "AT7_cross_document",
            (
                "single_turn_combination_"
                "violation_event"
            ),
        ),
        (
            "AT8 cumulative",
            "AT8_cumulative",
            (
                "cross_turn_cumulative_"
                "leakage_event"
            ),
        ),
        (
            "AT8 post-revocation",
            "AT8_revocation",
            (
                "post_revocation_"
                "redisclosure_event"
            ),
        ),
    ]

    for condition in [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    ]:
        row = {
            "Condition": (
                condition
            ),
        }

        for (
            label,
            scope,
            metric,
        ) in metrics:
            metric_row = find_binary(
                binary,
                condition,
                metric,
                scope,
            )

            row[
                label
            ] = (
                format_binary_row(
                    metric_row
                )
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def extract_ablation_row(
    path,
    label,
):
    df = read_csv(
        path
    )

    if df.empty:
        raise ValueError(
            f"No rows found in {path}"
        )

    row = df.iloc[0]

    return {
        "Configuration": (
            label
        ),
        "n": int(
            row["n"]
        ),
        "Forbidden disclosure rate": (
            as_percent(
                row[
                    "forbidden_disclosure_rate"
                ]
            )
        ),
        "Cumulative leak rate": (
            as_percent(
                row[
                    "new_cumulative_combination_leak_rate"
                ]
            )
        ),
        "Overblocking rate": (
            as_percent(
                row[
                    "overblocking_rate"
                ]
            )
        ),
        "Authorized fact recall": (
            as_percent(
                row[
                    "mean_authorized_fact_recall"
                ]
            )
        ),
        "Authorized utility retention": (
            as_percent(
                row[
                    "mean_authorized_utility_retention"
                ]
            )
        ),
        "Internal weighted utility retention": (
            as_percent(
                row[
                    "mean_internal_utility_retention"
                ]
            )
        ),
    }


def build_table_3_ablation():
    baseline_df = read_csv(
        PATHS[
            "qwen05_dev_summary"
        ]
    )

    baseline_subset = (
        baseline_df[
            baseline_df[
                "condition"
            ]
            == "F"
        ]
    )

    if baseline_subset.empty:
        raise ValueError(
            "Condition F was not found "
            "in Qwen 0.5B dev summary."
        )

    baseline = (
        baseline_subset.iloc[0]
    )

    rows = [
        {
            "Configuration": (
                "Full F"
            ),
            "n": int(
                baseline["n"]
            ),
            "Forbidden disclosure rate": (
                as_percent(
                    baseline[
                        "forbidden_disclosure_rate"
                    ]
                )
            ),
            "Cumulative leak rate": (
                as_percent(
                    baseline[
                        "new_cumulative_combination_leak_rate"
                    ]
                )
            ),
            "Overblocking rate": (
                as_percent(
                    baseline[
                        "overblocking_rate"
                    ]
                )
            ),
            "Authorized fact recall": (
                as_percent(
                    baseline[
                        "mean_authorized_fact_recall"
                    ]
                )
            ),
            "Authorized utility retention": (
                as_percent(
                    baseline[
                        "mean_authorized_utility_retention"
                    ]
                )
            ),
            "Internal weighted utility retention": (
                as_percent(
                    baseline[
                        "mean_internal_utility_retention"
                    ]
                )
            ),
        },
        extract_ablation_row(
            PATHS[
                "ablation_no_history"
            ],
            "F without history",
        ),
        extract_ablation_row(
            PATHS[
                "ablation_no_weight"
            ],
            "F without utility weights",
        ),
        extract_ablation_row(
            PATHS[
                "ablation_all_block"
            ],
            "F all-block",
        ),
    ]

    return pd.DataFrame(
        rows
    )


def crossmodel_scope_row(
    model_name,
    analysis_dir,
    condition,
):
    scope_binary = (
        read_analysis_file(
            analysis_dir,
            "scope_binary_summary.csv",
        )
    )

    at7 = find_binary(
        scope_binary,
        condition,
        (
            "single_turn_combination_"
            "violation_event"
        ),
        "AT7_cross_document",
    )

    at8 = find_binary(
        scope_binary,
        condition,
        (
            "cross_turn_cumulative_"
            "leakage_event"
        ),
        "AT8_cumulative",
    )

    revocation = find_binary(
        scope_binary,
        condition,
        (
            "post_revocation_"
            "redisclosure_event"
        ),
        "AT8_revocation",
    )

    return {
        "Model": (
            model_name
        ),
        "Condition": (
            condition
        ),
        "AT7 cross-document": (
            format_binary_row(
                at7
            )
        ),
        "AT8 cumulative": (
            format_binary_row(
                at8
            )
        ),
        "AT8 post-revocation": (
            format_binary_row(
                revocation
            )
        ),
    }


def build_table_4_crossmodel():
    model_sources = [
        (
            "Qwen2.5-0.5B",
            PATHS[
                "qwen05_test_analysis"
            ],
        ),
        (
            "Qwen2.5-3B",
            PATHS[
                "qwen3_test_analysis"
            ],
        ),
        (
            "Phi-3.5-mini",
            PATHS[
                "phi35_test_analysis"
            ],
        ),
    ]

    rows = []

    for (
        model_name,
        analysis_dir,
    ) in model_sources:
        for condition in [
            "C",
            "E",
            "F",
        ]:
            rows.append(
                crossmodel_scope_row(
                    model_name,
                    analysis_dir,
                    condition,
                )
            )

    return pd.DataFrame(
        rows
    )


def build_table_5_sensitivity():
    df = read_csv(
        PATHS[
            "utility_sensitivity"
        ]
    )

    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "Perturbation": (
                    f"±{float(row['perturbation_level']) * 100:.0f}%"
                ),
                "Trials": int(
                    row[
                        "n_trials"
                    ]
                ),
                "Safety preserved": (
                    as_percent(
                        row[
                            "safety_preserved_rate"
                        ]
                    )
                ),
                "Exact selection agreement": (
                    as_percent(
                        row[
                            "exact_selection_agreement_rate"
                        ]
                    )
                ),
                "Baseline-optimal equivalent": (
                    as_percent(
                        row[
                            "baseline_optimal_equivalent_rate"
                        ]
                    )
                ),
                "Changed but equivalent": (
                    as_percent(
                        row[
                            "selection_changed_but_equivalent_rate"
                        ]
                    )
                ),
                "Strict rank reversal": (
                    as_percent(
                        row[
                            "strict_rank_reversal_event_rate"
                        ]
                    )
                ),
                "Mean rank-reversal pair rate": (
                    as_percent(
                        row[
                            "mean_strict_rank_reversal_pair_rate"
                        ]
                    )
                ),
                "Mean baseline regret / total utility": (
                    as_percent(
                        row[
                            "mean_baseline_regret_relative_total"
                        ]
                    )
                ),
                "Mean original utility retention": (
                    as_percent(
                        row[
                            "mean_original_utility_retention"
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_table_6_paired_tests():
    model_sources = [
        (
            "Qwen2.5-0.5B",
            PATHS[
                "qwen05_test_analysis"
            ],
        ),
        (
            "Qwen2.5-3B",
            PATHS[
                "qwen3_test_analysis"
            ],
        ),
        (
            "Phi-3.5-mini",
            PATHS[
                "phi35_test_analysis"
            ],
        ),
    ]

    target_metrics = {
        (
            "single_turn_combination_"
            "violation_event"
        ): (
            "AT7 cross-document"
        ),
        (
            "cross_turn_cumulative_"
            "leakage_event"
        ): (
            "AT8 cumulative"
        ),
        (
            "post_revocation_"
            "redisclosure_event"
        ): (
            "AT8 post-revocation"
        ),
        (
            "forbidden_disclosure_event"
        ): (
            "Overall forbidden disclosure"
        ),
    }

    rows = []

    for (
        model_name,
        analysis_dir,
    ) in model_sources:
        df = read_analysis_file(
            analysis_dir,
            "paired_exact_tests.csv",
        )

        for _, row in df.iterrows():
            metric = row[
                "metric"
            ]

            if metric not in target_metrics:
                continue

            if row[
                "condition_b"
            ] != "F":
                continue

            rows.append(
                {
                    "Model": (
                        model_name
                    ),
                    "Metric": (
                        target_metrics[
                            metric
                        ]
                    ),
                    "Comparison": (
                        f"{row['condition_a']} vs "
                        f"{row['condition_b']}"
                    ),
                    "Paired n": int(
                        row[
                            "paired_n"
                        ]
                    ),
                    "A0/B1": int(
                        row[
                            "a0_b1"
                        ]
                    ),
                    "A1/B0": int(
                        row[
                            "a1_b0"
                        ]
                    ),
                    "Exact two-sided p": (
                        f"{float(row['exact_two_sided_p']):.6g}"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def dataframe_to_markdown(
    df,
):
    if df.empty:
        return "_No data_"

    headers = [
        str(column)
        for column
        in df.columns
    ]

    lines = []

    lines.append(
        "| "
        + " | ".join(
            headers
        )
        + " |"
    )

    lines.append(
        "| "
        + " | ".join(
            [
                "---"
                for _ in headers
            ]
        )
        + " |"
    )

    for _, row in df.iterrows():
        values = []

        for value in row:
            if pd.isna(value):
                value = ""

            value = (
                str(value)
                .replace(
                    "|",
                    "\\|",
                )
                .replace(
                    "\n",
                    " ",
                )
            )

            values.append(
                value
            )

        lines.append(
            "| "
            + " | ".join(
                values
            )
            + " |"
        )

    return "\n".join(
        lines
    )


def save_table(
    df,
    filename,
):
    path = (
        OUTPUT_DIR
        / filename
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    return path


def build_markdown_report(
    tables,
):
    sections = [
        (
            "# Final Paper Result Tables",
            None,
        ),
        (
            "## Table 1. "
            "Overall A–F security performance",
            tables[
                "table_1"
            ],
        ),
        (
            "## Table 2. "
            "AT7 and AT8 security performance",
            tables[
                "table_2"
            ],
        ),
        (
            "## Table 3. "
            "Ablation analysis",
            tables[
                "table_3"
            ],
        ),
        (
            "## Table 4. "
            "Cross-model robustness",
            tables[
                "table_4"
            ],
        ),
        (
            "## Table 5. "
            "Utility-weight sensitivity",
            tables[
                "table_5"
            ],
        ),
        (
            "## Table 6. "
            "Paired exact tests",
            tables[
                "table_6"
            ],
        ),
    ]

    parts = []

    for (
        heading,
        dataframe,
    ) in sections:
        parts.append(
            heading
        )

        if dataframe is not None:
            parts.append(
                dataframe_to_markdown(
                    dataframe
                )
            )

        parts.append(
            ""
        )

    parts.extend(
        [
            "## Interpretation cautions",
            "",
            (
                "- Qwen2.5-0.5B full-test results are the "
                "primary experiment."
            ),
            (
                "- Qwen2.5-3B and Phi-3.5-mini results are "
                "focused AT7/AT8 robustness checks rather "
                "than full A–F replications."
            ),
            (
                "- AT8 cumulative and post-revocation scopes "
                "contain eight evaluated second-turn cases per "
                "condition in the test split; confidence "
                "intervals are therefore wide."
            ),
            (
                "- Zero observed events should not be described "
                "as proof of zero deployment risk."
            ),
            (
                "- LLM generation latency is not used for "
                "cross-condition performance claims because "
                "the generation path differs when the proposed "
                "controller blocks or reduces response content."
            ),
            (
                "- Utility sensitivity intervals summarize "
                "Monte Carlo perturbation observations over "
                "the fixed synthetic rule set; they are not "
                "deployment-population confidence intervals."
            ),
            "",
        ]
    )

    return "\n".join(
        parts
    )


def write_manifest(
    generated_paths,
):
    manifest = {
        "purpose": (
            "Paper-ready consolidation of the final "
            "experiment results."
        ),
        "primary_model": (
            "Qwen/Qwen2.5-0.5B-Instruct"
        ),
        "cross_model_checks": [
            (
                "Qwen/Qwen2.5-3B-Instruct"
            ),
            (
                "microsoft/Phi-3.5-mini-instruct"
            ),
        ],
        "source_files": {
            key: str(
                value.relative_to(
                    BASE_DIR
                )
            )
            for (
                key,
                value,
            )
            in PATHS.items()
        },
        "generated_files": [
            str(
                path.relative_to(
                    BASE_DIR
                )
            )
            for path
            in generated_paths
        ],
        "notes": [
            (
                "Primary A-F claims should use the "
                "Qwen2.5-0.5B final test split."
            ),
            (
                "Ablation comparisons use the development "
                "split so that the held-out final test was "
                "not repeatedly optimized against."
            ),
            (
                "Cross-model checks are restricted to "
                "conditions C, E, and F on AT7 and AT8."
            ),
            (
                "Do not interpret observed zero-event rates "
                "as universal guarantees."
            ),
        ],
    }

    path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return path


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_1 = (
        build_table_1_overall_af()
    )

    table_2 = (
        build_table_2_at7_at8()
    )

    table_3 = (
        build_table_3_ablation()
    )

    table_4 = (
        build_table_4_crossmodel()
    )

    table_5 = (
        build_table_5_sensitivity()
    )

    table_6 = (
        build_table_6_paired_tests()
    )

    tables = {
        "table_1": table_1,
        "table_2": table_2,
        "table_3": table_3,
        "table_4": table_4,
        "table_5": table_5,
        "table_6": table_6,
    }

    generated_paths = []

    generated_paths.append(
        save_table(
            table_1,
            "table_1_overall_af.csv",
        )
    )

    generated_paths.append(
        save_table(
            table_2,
            "table_2_at7_at8.csv",
        )
    )

    generated_paths.append(
        save_table(
            table_3,
            "table_3_ablation.csv",
        )
    )

    generated_paths.append(
        save_table(
            table_4,
            "table_4_crossmodel.csv",
        )
    )

    generated_paths.append(
        save_table(
            table_5,
            "table_5_utility_sensitivity.csv",
        )
    )

    generated_paths.append(
        save_table(
            table_6,
            "table_6_paired_exact_tests.csv",
        )
    )

    markdown_report = (
        build_markdown_report(
            tables
        )
    )

    markdown_path = (
        OUTPUT_DIR
        / "paper_tables.md"
    )

    with open(
        markdown_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            markdown_report
        )

    generated_paths.append(
        markdown_path
    )

    manifest_path = (
        write_manifest(
            generated_paths
        )
    )

    generated_paths.append(
        manifest_path
    )

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL PAPER TABLES COMPLETE"
    )

    print(
        "=" * 72
    )

    for path in generated_paths:
        print(
            "-",
            path
        )

    print()
    print(
        "Primary result preview:"
    )

    print()
    print(
        table_1.to_string(
            index=False
        )
    )

    print()
    print(
        "Cross-model preview:"
    )

    print()
    print(
        table_4.to_string(
            index=False
        )
    )

    print()
    print(
        "Sensitivity preview:"
    )

    print()
    print(
        table_5.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()