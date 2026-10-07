import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


BINARY_METRICS = [
    "unauthorized_evidence_emission_event",
    "unauthorized_retrieval_exposure_event",
    "forbidden_disclosure_event",
    "single_turn_combination_violation_event",
    "cross_turn_cumulative_leakage_event",
    "post_revocation_redisclosure_event",
    "overblocking_event",
    "hallucination_event",
    "guard_triggered",
]

CONTINUOUS_METRICS = [
    "authorized_fact_recall",
    "authorized_utility_retention_rate",
    "internal_utility_retention_rate",
    "retrieval_sec",
    "control_sec",
    "generation_sec",
    "guard_sec",
    "total_sec",
]


def to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan

    p = successes / n
    denominator = 1.0 + (z * z / n)
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(
            (p * (1.0 - p) / n)
            + (z * z / (4.0 * n * n))
        )
        / denominator
    )
    return center - half, center + half


def mean_ci(values, z=1.959963984540054):
    values = pd.Series(values).dropna().astype(float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    if n == 1:
        return mean, np.nan, np.nan
    se = float(values.std(ddof=1) / math.sqrt(n))
    return mean, mean - z * se, mean + z * se


def exact_paired_binary_pvalue(x, y):
    """
    Exact two-sided McNemar/binomial sign test using discordant pairs only.
    Returns n01, n10, p_value.
    """
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if pairs.empty:
        return 0, 0, np.nan

    pairs["x"] = pairs["x"].astype(int)
    pairs["y"] = pairs["y"].astype(int)

    n01 = int(((pairs["x"] == 0) & (pairs["y"] == 1)).sum())
    n10 = int(((pairs["x"] == 1) & (pairs["y"] == 0)).sum())
    n = n01 + n10

    if n == 0:
        return n01, n10, 1.0

    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = min(1.0, 2.0 * tail)
    return n01, n10, p_value


def normalize_metrics(df):
    required = {
        "scenario_id",
        "condition",
        "scenario_type",
        "turn",
        "attack_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["turn"] = to_numeric(df["turn"])

    if "direct_unauthorized_leakage_event" in df.columns:
        df["unauthorized_evidence_emission_event"] = to_numeric(
            df["direct_unauthorized_leakage_event"]
        )
    else:
        df["unauthorized_evidence_emission_event"] = np.nan

    if "unauthorized_target_document_retrieval_event" in df.columns:
        df["unauthorized_retrieval_exposure_event"] = to_numeric(
            df["unauthorized_target_document_retrieval_event"]
        )
    else:
        df["unauthorized_retrieval_exposure_event"] = np.nan

    combo = to_numeric(df.get("combination_violation_event", pd.Series(np.nan, index=df.index)))
    cumulative = to_numeric(
        df.get("new_cumulative_combination_leak_event", pd.Series(np.nan, index=df.index))
    )

    df["single_turn_combination_violation_event"] = np.where(
        df["scenario_type"].eq("cross_document_inference"),
        combo,
        np.nan,
    )

    df["cross_turn_cumulative_leakage_event"] = np.where(
        df["scenario_type"].eq("multiturn_cumulative") & df["turn"].gt(1),
        cumulative,
        np.nan,
    )

    if "post_revocation_redisclosure_event" in df.columns:
        revocation = to_numeric(df["post_revocation_redisclosure_event"])
    else:
        revocation = pd.Series(np.nan, index=df.index)

    df["post_revocation_redisclosure_event"] = np.where(
        df["scenario_type"].eq("revocation_multiturn") & df["turn"].gt(1),
        revocation,
        np.nan,
    )

    for column in [
        "forbidden_disclosure_event",
        "overblocking_event",
        "hallucination_event",
        "guard_triggered",
        "authorized_fact_recall",
        "authorized_utility_retention_rate",
        "internal_utility_retention_rate",
        "retrieval_sec",
        "control_sec",
        "generation_sec",
        "guard_sec",
        "total_sec",
    ]:
        if column in df.columns:
            df[column] = to_numeric(df[column])
        else:
            df[column] = np.nan

    return df


def binary_summary(df, group_columns):
    rows = []

    grouped = df.groupby(group_columns, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        prefix = dict(zip(group_columns, group_key))

        for metric in BINARY_METRICS:
            values = group[metric].dropna()
            n = len(values)
            if n == 0:
                continue

            successes = int(values.sum())
            rate = float(values.mean())
            low, high = wilson_interval(successes, n)

            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "n": n,
                    "events": successes,
                    "rate": rate,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    return pd.DataFrame(rows)


def continuous_summary(df, group_columns):
    rows = []

    grouped = df.groupby(group_columns, dropna=False)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        prefix = dict(zip(group_columns, group_key))

        for metric in CONTINUOUS_METRICS:
            values = group[metric].dropna()
            if len(values) == 0:
                continue

            mean, low, high = mean_ci(values)
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "n": len(values),
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    return pd.DataFrame(rows)


def paired_tests(df):
    comparisons = [
        ("C", "E"),
        ("C", "F"),
        ("E", "F"),
    ]

    metric_scopes = {
        "unauthorized_evidence_emission_event": df.index,
        "unauthorized_retrieval_exposure_event": df.index,
        "forbidden_disclosure_event": df.index,
        "single_turn_combination_violation_event": df[
            "scenario_type"
        ].eq("cross_document_inference"),
        "cross_turn_cumulative_leakage_event": (
            df["scenario_type"].eq("multiturn_cumulative")
            & df["turn"].gt(1)
        ),
        "post_revocation_redisclosure_event": (
            df["scenario_type"].eq("revocation_multiturn")
            & df["turn"].gt(1)
        ),
    }

    rows = []

    for metric, scope in metric_scopes.items():
        subset = df.loc[scope, ["scenario_id", "condition", metric]].copy()
        subset = subset.dropna(subset=[metric])
        if subset.empty:
            continue

        pivot = subset.pivot_table(
            index="scenario_id",
            columns="condition",
            values=metric,
            aggfunc="first",
        )

        for condition_a, condition_b in comparisons:
            if condition_a not in pivot.columns or condition_b not in pivot.columns:
                continue

            paired = pivot[[condition_a, condition_b]].dropna()
            if paired.empty:
                continue

            n01, n10, p_value = exact_paired_binary_pvalue(
                paired[condition_a],
                paired[condition_b],
            )

            rows.append(
                {
                    "metric": metric,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "paired_n": len(paired),
                    "a0_b1": n01,
                    "a1_b0": n10,
                    "exact_two_sided_p": p_value,
                }
            )

    return pd.DataFrame(rows)


def make_scope_label(df):
    labels = np.full(len(df), "other", dtype=object)
    labels[df["scenario_type"].eq("normal_allowed")] = "normal_allowed"
    labels[df["scenario_type"].eq("normal_denied")] = "normal_denied"
    labels[df["scenario_type"].eq("adversarial_single_turn")] = "AT1_AT6"
    labels[df["scenario_type"].eq("cross_document_inference")] = "AT7_cross_document"
    labels[df["scenario_type"].eq("multiturn_cumulative")] = "AT8_cumulative"
    labels[df["scenario_type"].eq("revocation_multiturn")] = "AT8_revocation"
    return labels


def analyze(input_csv, output_dir):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = normalize_metrics(df)
    df["analysis_scope"] = make_scope_label(df)

    normalized_path = output_dir / "normalized_results.csv"
    overall_binary_path = output_dir / "overall_binary_summary.csv"
    overall_continuous_path = output_dir / "overall_continuous_summary.csv"
    scope_binary_path = output_dir / "scope_binary_summary.csv"
    scope_continuous_path = output_dir / "scope_continuous_summary.csv"
    paired_path = output_dir / "paired_exact_tests.csv"

    df.to_csv(normalized_path, index=False, encoding="utf-8-sig")

    overall_binary = binary_summary(df, ["condition"])
    overall_continuous = continuous_summary(df, ["condition"])
    scope_binary = binary_summary(df, ["condition", "analysis_scope"])
    scope_continuous = continuous_summary(df, ["condition", "analysis_scope"])
    paired = paired_tests(df)

    overall_binary.to_csv(overall_binary_path, index=False, encoding="utf-8-sig")
    overall_continuous.to_csv(overall_continuous_path, index=False, encoding="utf-8-sig")
    scope_binary.to_csv(scope_binary_path, index=False, encoding="utf-8-sig")
    scope_continuous.to_csv(scope_continuous_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")

    print("=" * 72)
    print("SECURITY RESULT ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Input: {input_csv}")
    print(f"Rows: {len(df)}")
    print(f"Conditions: {sorted(df['condition'].dropna().unique().tolist())}")
    print()
    print("Generated:")
    for path in [
        normalized_path,
        overall_binary_path,
        overall_continuous_path,
        scope_binary_path,
        scope_continuous_path,
        paired_path,
    ]:
        print(f"- {path}")

    key_metrics = overall_binary[
        overall_binary["metric"].isin(
            [
                "unauthorized_evidence_emission_event",
                "unauthorized_retrieval_exposure_event",
                "forbidden_disclosure_event",
            ]
        )
    ]
    if not key_metrics.empty:
        print("\nKey overall binary metrics:")
        print(key_metrics.to_string(index=False))

    at7 = scope_binary[
        (scope_binary["analysis_scope"] == "AT7_cross_document")
        & (
            scope_binary["metric"]
            == "single_turn_combination_violation_event"
        )
    ]
    if not at7.empty:
        print("\nAT7 single-turn combination violation:")
        print(at7.to_string(index=False))

    at8 = scope_binary[
        (scope_binary["analysis_scope"] == "AT8_cumulative")
        & (
            scope_binary["metric"]
            == "cross_turn_cumulative_leakage_event"
        )
    ]
    if not at8.empty:
        print("\nAT8 cross-turn cumulative leakage:")
        print(at8.to_string(index=False))

    revocation = scope_binary[
        (scope_binary["analysis_scope"] == "AT8_revocation")
        & (
            scope_binary["metric"]
            == "post_revocation_redisclosure_event"
        )
    ]
    if not revocation.empty:
        print("\nAT8 post-revocation redisclosure:")
        print(revocation.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-label and separate security metrics, calculate 95% confidence "
            "intervals, and run paired exact tests for C/E/F."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Detail result CSV, e.g. results/full_test_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <input_parent>/analysis_<input_stem>",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_path.parent / f"analysis_{input_path.stem}"
    )
    analyze(input_path, output_dir)


if __name__ == "__main__":
    main()
