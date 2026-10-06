import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_manager import DatasetManager


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FULL_DATA_DIR = DATA_DIR / "full_experiment"
RESULT_DIR = BASE_DIR / "results"

SCENARIO_FILE = (
    FULL_DATA_DIR
    / "scenarios.json"
)

FACT_UTILITY_FILE = (
    FULL_DATA_DIR
    / "fact_utility.json"
)


DEFAULT_CONDITIONS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F"
]


SMOKE_SCENARIO_IDS = [
    "NA-001",
    "ND-001",
    "AT1-001",
    "AT4-001",
    "AT7-001",
    "AT8-C-001-T1",
    "AT8-C-001-T2"
]


# ============================================================
# JSON
# ============================================================

def load_json(
    path
):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


def split_pipe(
    value
):

    if value is None:
        return []

    if (
        isinstance(
            value,
            float
        )
        and np.isnan(
            value
        )
    ):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    return [
        item
        for item
        in text.split("|")
        if item
    ]


# ============================================================
# Full Dataset Activation
# ============================================================

def ensure_full_dataset_active():

    required = [
        "users.json",
        "policies.json",
        "documents.json",
        "inference_rules.json",
        "fact_utility.json",
        "temporary_authorizations.json",
        "scenarios.json"
    ]

    missing = [
        filename
        for filename
        in required
        if not (
            FULL_DATA_DIR
            / filename
        ).exists()
    ]

    if missing:

        raise FileNotFoundError(
            "data/full_experiment에 "
            "필요한 파일이 없습니다: "
            + ", ".join(
                missing
            )
        )

    manager = (
        DatasetManager()
    )

    if (
        manager.detect_dataset()
        != "full_experiment"
    ):

        print(
            "Activating full experiment dataset..."
        )

        manager.activate_full()

    else:

        manager.write_state(
            "full_experiment"
        )


# ------------------------------------------------------------
# 매우 중요
#
# PilotExperimentRunnerV2를 import하기 전에
# full dataset을 data/에 활성화한다.
#
# 그렇지 않으면 WeightedControlledRAG가
# 기존 10문서 pilot 데이터를 먼저 읽을 수 있다.
# ------------------------------------------------------------

ensure_full_dataset_active()


# ============================================================
# Imports after dataset activation
# ============================================================

from src.pilot_experiment_runner_v2 import (
    PilotExperimentRunnerV2
)

from src.weighted_controlled_rag import (
    WeightedControlledRAG
)

from src.evaluation_metrics import (
    SecurityEvaluationMetrics
)


# ============================================================
# Full Experiment Runner
# ============================================================

class FullExperimentRunner(
    PilotExperimentRunnerV2
):

    def __init__(
        self,
        split="dev",
        conditions=None,
        smoke=False,
        allow_test=False,
        run_name=None,
        checkpoint_every=10
    ):

        # ----------------------------------------------------
        # Split
        # ----------------------------------------------------

        if split not in {
            "dev",
            "test"
        }:

            raise ValueError(
                "split은 'dev' 또는 "
                "'test'여야 합니다."
            )

        # Test는 실수로 돌리지 못하게 잠금
        if (
            split == "test"
            and not allow_test
        ):

            raise RuntimeError(
                "test split은 잠겨 있습니다. "
                "최종 실험에서만 "
                "--allow-test를 사용하세요."
            )

        # ----------------------------------------------------
        # Conditions
        # ----------------------------------------------------

        self.CONDITIONS = list(
            conditions
            or DEFAULT_CONDITIONS
        )

        invalid = [
            condition
            for condition
            in self.CONDITIONS
            if condition
            not in DEFAULT_CONDITIONS
        ]

        if invalid:

            raise ValueError(
                "알 수 없는 조건: "
                + ", ".join(
                    invalid
                )
            )

        self.split = split
        self.smoke = bool(
            smoke
        )

        self.checkpoint_every = max(
            1,
            int(
                checkpoint_every
            )
        )

        # ----------------------------------------------------
        # Result name
        # ----------------------------------------------------

        if run_name is None:

            run_name = (
                f"full_{split}"
            )

            if smoke:

                run_name += (
                    "_smoke"
                )

        self.run_name = (
            run_name
        )

        RESULT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.detail_result_file = (
            RESULT_DIR
            / f"{run_name}_results.csv"
        )

        self.summary_result_file = (
            RESULT_DIR
            / f"{run_name}_summary.csv"
        )

        self.attack_summary_file = (
            RESULT_DIR
            / f"{run_name}_attack_summary.csv"
        )

        # ====================================================
        # Scenario Load
        # ====================================================

        all_scenarios = (
            load_json(
                SCENARIO_FILE
            )
        )

        self.scenarios = [
            scenario
            for scenario
            in all_scenarios
            if (
                scenario.get(
                    "split"
                )
                == split
            )
        ]

        # ----------------------------------------------------
        # Smoke Test
        # ----------------------------------------------------

        if smoke:

            if split != "dev":

                raise ValueError(
                    "--smoke는 "
                    "dev에서만 "
                    "사용할 수 있습니다."
                )

            scenario_map = {
                scenario[
                    "scenario_id"
                ]:
                    scenario
                for scenario
                in self.scenarios
            }

            missing = [
                scenario_id
                for scenario_id
                in SMOKE_SCENARIO_IDS
                if scenario_id
                not in scenario_map
            ]

            if missing:

                raise ValueError(
                    "Smoke scenario 누락: "
                    + ", ".join(
                        missing
                    )
                )

            self.scenarios = [
                scenario_map[
                    scenario_id
                ]
                for scenario_id
                in SMOKE_SCENARIO_IDS
            ]

        if not self.scenarios:

            raise ValueError(
                f"{split} split에 "
                "시나리오가 없습니다."
            )

        self.scenario_map = {
            scenario[
                "scenario_id"
            ]:
                scenario
            for scenario
            in self.scenarios
        }

        # ====================================================
        # User / Policy
        # ====================================================

        self.users = (
            load_json(
                DATA_DIR
                / "users.json"
            )
        )

        self.policies = (
            load_json(
                DATA_DIR
                / "policies.json"
            )
        )

        self.user_map = {
            user[
                "user_id"
            ]:
                user
            for user
            in self.users
        }

        # ====================================================
        # Validation
        # ====================================================

        self._validate_active_dataset()

        self._validate_session_order()

        # ====================================================
        # RAG
        # ====================================================

        print(
            "Loading full experiment "
            "RAG pipeline..."
        )

        self.rag = (
            WeightedControlledRAG()
        )

        self.documents = (
            self.rag.documents
        )

        self.document_map = {
            doc[
                "document_id"
            ]:
                doc
            for doc
            in self.documents
        }

        self.guard = (
            self.rag.output_guard
        )

        self.controller = (
            self.rag.controller
        )

        # ====================================================
        # Metrics
        # ====================================================

        self.metrics = (
            SecurityEvaluationMetrics(
                documents=
                    self.documents,

                guard=
                    self.guard,

                controller=
                    self.controller
            )
        )

        # ====================================================
        # Unrestricted Retrieval
        # ====================================================

        print(
            "Building unrestricted "
            "retrieval index..."
        )

        self.unrestricted_document_ids = [
            doc[
                "document_id"
            ]
            for doc
            in self.documents
        ]

        unrestricted_texts = [
            self.document_to_text(
                doc
            )
            for doc
            in self.documents
        ]

        self.unrestricted_embeddings = (
            self.rag
            .retriever
            .model
            .encode(
                unrestricted_texts,
                normalize_embeddings=True
            )
        )

        # ====================================================
        # Authorized Documents
        # ====================================================

        self.authorized_documents = (
            self.build_authorized_document_map()
        )

        # ====================================================
        # Oracle
        # ====================================================

        self.oracle_results = (
            self.build_oracle_results()
        )

        # ====================================================
        # A-E history
        # ====================================================

        self.histories = {
            condition: {}
            for condition
            in [
                "A",
                "B",
                "C",
                "D",
                "E"
            ]
            if condition
            in self.CONDITIONS
        }

        # ====================================================
        # Observed facts
        # ====================================================

        self.observed_facts = {
            condition: {}
            for condition
            in self.CONDITIONS
        }

        self.rows = []

        # ====================================================
        # Utility
        # ====================================================

        self.fact_utility_map = (
            self._load_fact_utility_map()
        )

        # ====================================================
        # Ready
        # ====================================================

        print()

        print(
            "=" * 72
        )

        print(
            "FULL EXPERIMENT RUNNER READY"
        )

        print(
            "=" * 72
        )

        print(
            "Split:",
            self.split
        )

        print(
            "Smoke:",
            self.smoke
        )

        print(
            "Conditions:",
            self.CONDITIONS
        )

        print(
            "Scenarios:",
            len(
                self.scenarios
            )
        )

        print(
            "Planned runs:",
            len(
                self.scenarios
            )
            * len(
                self.CONDITIONS
            )
        )

        print(
            "Documents:",
            len(
                self.documents
            )
        )

        print(
            "Users:",
            len(
                self.users
            )
        )

        print(
            "=" * 72
        )

    # ========================================================
    # Validation
    # ========================================================

    def _validate_active_dataset(
        self
    ):

        if (
            len(
                self.users
            )
            != 12
        ):

            raise RuntimeError(
                "활성 사용자 수 오류: "
                f"{len(self.users)}"
            )

        expected = {
            "G01",
            "G02",
            "G03",

            "P01",
            "P02",
            "P03",

            "O01",
            "O02",
            "O03",

            "S01",
            "S02",
            "S03"
        }

        if (
            set(
                self.user_map
            )
            != expected
        ):

            raise RuntimeError(
                "활성 사용자 ID가 "
                "full dataset과 "
                "일치하지 않습니다."
            )

    def _validate_session_order(
        self
    ):

        last_turn = {}

        for scenario in (
            self.scenarios
        ):

            session_id = (
                scenario[
                    "session_id"
                ]
            )

            turn = int(
                scenario[
                    "turn"
                ]
            )

            if (
                session_id
                in last_turn
                and turn
                < last_turn[
                    session_id
                ]
            ):

                raise RuntimeError(
                    "세션 turn 순서 오류: "
                    f"{session_id}"
                )

            last_turn[
                session_id
            ] = turn

    # ========================================================
    # Utility
    # ========================================================

    def _load_fact_utility_map(
        self
    ):

        raw = load_json(
            FACT_UTILITY_FILE
        )

        result = {}

        # ----------------------------------------------------
        # Dict format
        # ----------------------------------------------------

        if isinstance(
            raw,
            dict
        ):

            for (
                fact_id,
                value
            ) in raw.items():

                if isinstance(
                    value,
                    (
                        int,
                        float
                    )
                ):

                    result[
                        fact_id
                    ] = float(
                        value
                    )

                elif isinstance(
                    value,
                    dict
                ):

                    utility = (
                        value.get(
                            "utility",
                            value.get(
                                "value",
                                value.get(
                                    "weight"
                                )
                            )
                        )
                    )

                    if (
                        utility
                        is not None
                    ):

                        result[
                            fact_id
                        ] = float(
                            utility
                        )

        # ----------------------------------------------------
        # List format
        # ----------------------------------------------------

        elif isinstance(
            raw,
            list
        ):

            for item in raw:

                if not isinstance(
                    item,
                    dict
                ):

                    continue

                fact_id = (
                    item.get(
                        "fact_id"
                    )
                )

                utility = (
                    item.get(
                        "utility",
                        item.get(
                            "value",
                            item.get(
                                "weight"
                            )
                        )
                    )
                )

                if (
                    fact_id
                    is not None
                    and utility
                    is not None
                ):

                    result[
                        fact_id
                    ] = float(
                        utility
                    )

        return result

    def _utility_sum(
        self,
        fact_ids
    ):

        return sum(
            self.fact_utility_map.get(
                fact_id,
                0.0
            )
            for fact_id
            in fact_ids
        )

    # ========================================================
    # Detail Result Extension
    # ========================================================

    def _enrich_detail_df(
        self,
        detail_df
    ):

        enriched = []

        for (
            _,
            row
        ) in detail_df.iterrows():

            item = (
                row.to_dict()
            )

            scenario = (
                self.scenario_map[
                    item[
                        "scenario_id"
                    ]
                ]
            )

            # ------------------------------------------------
            # Scenario metadata
            # ------------------------------------------------

            target_docs = (
                scenario.get(
                    "target_document_ids",
                    []
                )
            )

            target_facts = (
                scenario.get(
                    "target_fact_ids",
                    []
                )
            )

            combo_rules = (
                scenario.get(
                    "combination_rule_ids",
                    []
                )
            )

            item[
                "split"
            ] = (
                self.split
            )

            item[
                "attack_type"
            ] = (
                scenario.get(
                    "attack_type",
                    "none"
                )
            )

            item[
                "query_variant"
            ] = (
                scenario.get(
                    "query_variant",
                    ""
                )
            )

            item[
                "target_documents"
            ] = "|".join(
                target_docs
            )

            item[
                "target_facts"
            ] = "|".join(
                target_facts
            )

            item[
                "combination_rules"
            ] = "|".join(
                combo_rules
            )

            # ------------------------------------------------
            # Retrieval diagnostics
            # ------------------------------------------------

            user_id = (
                item[
                    "user_id"
                ]
            )

            retrieved_ids = set(
                split_pipe(
                    item.get(
                        "retrieved_documents"
                    )
                )
            )

            authorized_target_docs = [
                document_id
                for document_id
                in target_docs
                if document_id
                in self.authorized_documents[
                    user_id
                ]
            ]

            unauthorized_target_docs = [
                document_id
                for document_id
                in target_docs
                if document_id
                not in self.authorized_documents[
                    user_id
                ]
            ]

            if authorized_target_docs:

                item[
                    "authorized_target_document_retrieval_rate"
                ] = (
                    len(
                        set(
                            authorized_target_docs
                        )
                        & retrieved_ids
                    )
                    / len(
                        authorized_target_docs
                    )
                )

            else:

                item[
                    "authorized_target_document_retrieval_rate"
                ] = np.nan

            item[
                "unauthorized_target_document_retrieval_event"
            ] = int(
                bool(
                    set(
                        unauthorized_target_docs
                    )
                    & retrieved_ids
                )
            )

            # ------------------------------------------------
            # F utility reconstruction
            # ------------------------------------------------

            allowed_facts = (
                split_pipe(
                    item.get(
                        "internal_allowed_facts"
                    )
                )
            )

            removed_facts = (
                split_pipe(
                    item.get(
                        "internal_removed_facts"
                    )
                )
            )

            allowed_utility = (
                self._utility_sum(
                    allowed_facts
                )
            )

            removed_utility = (
                self._utility_sum(
                    removed_facts
                )
            )

            candidate_utility = (
                allowed_utility
                + removed_utility
            )

            if (
                item[
                    "condition"
                ]
                == "F"
            ):

                item[
                    "internal_candidate_utility"
                ] = (
                    candidate_utility
                )

                item[
                    "internal_removed_utility"
                ] = (
                    removed_utility
                )

                item[
                    "internal_retained_utility"
                ] = (
                    allowed_utility
                )

                if (
                    candidate_utility
                    > 0
                ):

                    item[
                        "internal_utility_retention_rate"
                    ] = (
                        allowed_utility
                        / candidate_utility
                    )

                else:

                    item[
                        "internal_utility_retention_rate"
                    ] = np.nan

            else:

                item[
                    "internal_candidate_utility"
                ] = np.nan

                item[
                    "internal_removed_utility"
                ] = np.nan

                item[
                    "internal_retained_utility"
                ] = np.nan

                item[
                    "internal_utility_retention_rate"
                ] = np.nan

            enriched.append(
                item
            )

        return pd.DataFrame(
            enriched
        )

    # ========================================================
    # Run
    # ========================================================

    def run(
        self
    ):

        total_planned = (
            len(
                self.scenarios
            )
            * len(
                self.CONDITIONS
            )
        )

        completed = 0

        print()

        print(
            "=" * 72
        )

        print(
            "FULL A-F EXPERIMENT START"
        )

        print(
            "=" * 72
        )

        try:

            for (
                scenario_index,
                scenario
            ) in enumerate(
                self.scenarios,
                start=1
            ):

                print()

                print(
                    f"[{scenario_index}/"
                    f"{len(self.scenarios)}] "
                    f"{scenario['scenario_id']} "
                    f"- "
                    f"{scenario['scenario_type']} "
                    f"- "
                    f"{scenario.get('attack_type', 'none')}"
                )

                oracle = (
                    self.oracle_results[
                        scenario[
                            "scenario_id"
                        ]
                    ]
                )

                print(
                    "  Oracle Allowed:",
                    oracle[
                        "allowed_fact_ids"
                    ]
                )

                print(
                    "  Oracle Blocked:",
                    oracle[
                        "blocked_fact_ids"
                    ]
                )

                for condition in (
                    self.CONDITIONS
                ):

                    print(
                        f"  Running {condition}..."
                    )

                    if condition == "F":

                        result = (
                            self.run_condition_f(
                                scenario
                            )
                        )

                    else:

                        result = (
                            self.run_baseline_condition(
                                condition=
                                    condition,

                                scenario=
                                    scenario
                            )
                        )

                    row = (
                        self.evaluate_result(
                            condition=
                                condition,

                            scenario=
                                scenario,

                            result=
                                result
                        )
                    )

                    self.rows.append(
                        row
                    )

                    completed += 1

                    print(
                        "    "
                        f"{completed}/"
                        f"{total_planned} "
                        f"direct="
                        f"{row['direct_unauthorized_leakage_event']}, "
                        f"forbidden="
                        f"{row['forbidden_disclosure_event']}, "
                        f"combo="
                        f"{row['combination_violation_event']}, "
                        f"cumulative="
                        f"{row['new_cumulative_combination_leak_event']}, "
                        f"overblock="
                        f"{row['overblocking_event']}, "
                        f"recall="
                        f"{row['authorized_fact_recall']}, "
                        f"guard="
                        f"{row['guard_triggered']}"
                    )

                    if (
                        completed
                        % self.checkpoint_every
                        == 0
                    ):

                        self.save_results(
                            checkpoint=True
                        )

        except Exception:

            print()

            print(
                "ERROR 발생. "
                "현재까지 결과를 저장합니다."
            )

            if self.rows:

                self.save_results(
                    checkpoint=True
                )

            raise

        self.save_results(
            checkpoint=False
        )

    # ========================================================
    # Summary
    # ========================================================

    @staticmethod
    def build_summary(
        detail_df,
        group_columns
    ):

        return (
            detail_df
            .groupby(
                group_columns,
                as_index=False,
                dropna=False
            )
            .agg(
                n=(
                    "scenario_id",
                    "count"
                ),

                direct_unauthorized_leak_rate=(
                    "direct_unauthorized_leakage_event",
                    "mean"
                ),

                mean_unauthorized_evidence_token_rate=(
                    "unauthorized_evidence_token_rate",
                    "mean"
                ),

                forbidden_disclosure_rate=(
                    "forbidden_disclosure_event",
                    "mean"
                ),

                mean_forbidden_target_disclosure_rate=(
                    "forbidden_target_disclosure_rate",
                    "mean"
                ),

                combination_violation_rate=(
                    "combination_violation_event",
                    "mean"
                ),

                new_cumulative_combination_leak_rate=(
                    "new_cumulative_combination_leak_event",
                    "mean"
                ),

                post_revocation_redisclosure_rate=(
                    "post_revocation_redisclosure_event",
                    "mean"
                ),

                overblocking_rate=(
                    "overblocking_event",
                    "mean"
                ),

                mean_authorized_fact_recall=(
                    "authorized_fact_recall",
                    "mean"
                ),

                mean_authorized_utility_retention=(
                    "authorized_utility_retention_rate",
                    "mean"
                ),

                mean_internal_utility_retention=(
                    "internal_utility_retention_rate",
                    "mean"
                ),

                mean_authorized_target_document_retrieval=(
                    "authorized_target_document_retrieval_rate",
                    "mean"
                ),

                unauthorized_target_document_retrieval_rate=(
                    "unauthorized_target_document_retrieval_event",
                    "mean"
                ),

                hallucination_rate=(
                    "hallucination_event",
                    "mean"
                ),

                guard_trigger_rate=(
                    "guard_triggered",
                    "mean"
                ),

                mean_retrieval_sec=(
                    "retrieval_sec",
                    "mean"
                ),

                mean_control_sec=(
                    "control_sec",
                    "mean"
                ),

                mean_generation_sec=(
                    "generation_sec",
                    "mean"
                ),

                mean_guard_sec=(
                    "guard_sec",
                    "mean"
                ),

                mean_total_sec=(
                    "total_sec",
                    "mean"
                )
            )
        )

    # ========================================================
    # Save
    # ========================================================

    def save_results(
        self,
        checkpoint=False
    ):

        if not self.rows:
            return

        detail_df = (
            pd.DataFrame(
                self.rows
            )
        )

        detail_df = (
            self._enrich_detail_df(
                detail_df
            )
        )

        detail_df.to_csv(
            self.detail_result_file,
            index=False,
            encoding="utf-8-sig"
        )

        summary_df = (
            self.build_summary(
                detail_df,
                [
                    "condition"
                ]
            )
        )

        summary_df.to_csv(
            self.summary_result_file,
            index=False,
            encoding="utf-8-sig"
        )

        attack_summary_df = (
            self.build_summary(
                detail_df,
                [
                    "condition",
                    "attack_type",
                    "scenario_type"
                ]
            )
        )

        attack_summary_df.to_csv(
            self.attack_summary_file,
            index=False,
            encoding="utf-8-sig"
        )

        if checkpoint:

            print(
                f"    checkpoint -> "
                f"{self.detail_result_file.name}"
            )

            return

        print()

        print(
            "=" * 72
        )

        print(
            "EXPERIMENT COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            "DETAIL:",
            self.detail_result_file
        )

        print(
            "SUMMARY:",
            self.summary_result_file
        )

        print(
            "ATTACK SUMMARY:",
            self.attack_summary_file
        )

        print()

        print(
            summary_df.to_string(
                index=False
            )
        )


# ============================================================
# Preflight
# ============================================================

def preflight(
    split,
    conditions,
    allow_test=False
):

    if (
        split == "test"
        and not allow_test
    ):

        raise RuntimeError(
            "test split은 잠겨 있습니다. "
            "--allow-test가 필요합니다."
        )

    scenarios = (
        load_json(
            SCENARIO_FILE
        )
    )

    selected = [
        scenario
        for scenario
        in scenarios
        if (
            scenario.get(
                "split"
            )
            == split
        )
    ]

    print(
        "=" * 72
    )

    print(
        "FULL EXPERIMENT PREFLIGHT"
    )

    print(
        "=" * 72
    )

    print(
        "All scenarios:",
        len(
            scenarios
        )
    )

    print(
        f"{split} scenarios:",
        len(
            selected
        )
    )

    print(
        "Conditions:",
        conditions
    )

    print(
        "Expected runs:",
        len(
            selected
        )
        * len(
            conditions
        )
    )

    print()

    print(
        "Attack distribution:"
    )

    print(
        pd.Series(
            [
                scenario.get(
                    "attack_type",
                    "none"
                )
                for scenario
                in selected
            ]
        )
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Full synthetic "
                "A-F experiment runner"
            )
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "test"
        ],
        default="dev"
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=
            DEFAULT_CONDITIONS,
        default=
            DEFAULT_CONDITIONS
    )

    parser.add_argument(
        "--smoke",
        action="store_true"
    )

    parser.add_argument(
        "--allow-test",
        action="store_true"
    )

    parser.add_argument(
        "--run-name",
        default=None
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10
    )

    parser.add_argument(
        "--preflight-only",
        action="store_true"
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = (
        parse_args()
    )

    if (
        args.preflight_only
    ):

        preflight(
            split=
                args.split,

            conditions=
                args.conditions,

            allow_test=
                args.allow_test
        )

        return

    runner = (
        FullExperimentRunner(
            split=
                args.split,

            conditions=
                args.conditions,

            smoke=
                args.smoke,

            allow_test=
                args.allow_test,

            run_name=
                args.run_name,

            checkpoint_every=
                args.checkpoint_every
        )
    )

    runner.run()


if __name__ == "__main__":

    main()
