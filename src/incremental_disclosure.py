import json
from itertools import combinations
from pathlib import Path

from src.dynamic_authorization import (
    DynamicAuthorizationManager
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_json(filename):

    with open(
        DATA_DIR / filename,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


class IncrementalDisclosureController:
    """
    과거 공개 Fact와 현재 후보 Fact의 조합을 검사하고,
    금지된 정보조합을 모두 차단하면서
    제거되는 업무가치의 총합을 최소화한다.

    추가 기능:
    동적 임시 권한의 유효시간을 반영한다.
    """

    def __init__(self):

        self.users = load_json(
            "users.json"
        )

        self.rules = load_json(
            "inference_rules.json"
        )

        self.fact_utility = load_json(
            "fact_utility.json"
        )

        self.user_map = {
            user["user_id"]: user
            for user in self.users
        }

        self.utility_map = {
            item["fact_id"]:
                float(
                    item[
                        "business_value"
                    ]
                )
            for item in self.fact_utility
        }

        self.dynamic_authorization = (
            DynamicAuthorizationManager()
        )

        self.validate_utility_values()

    # =================================================
    # Utility 검증
    # =================================================

    def validate_utility_values(self):

        required_fact_ids = set()

        for rule in self.rules:

            required_fact_ids.update(
                rule[
                    "required_facts"
                ]
            )

        missing = (
            required_fact_ids
            - set(
                self.utility_map.keys()
            )
        )

        if missing:

            raise ValueError(
                "업무가치가 정의되지 않은 "
                "Fact가 있습니다: "
                f"{sorted(missing)}"
            )

    # =================================================
    # 사용자
    # =================================================

    def get_user(
        self,
        user_id
    ):

        if user_id not in self.user_map:

            raise ValueError(
                f"Unknown user: {user_id}"
            )

        return self.user_map[
            user_id
        ]

    # =================================================
    # Business Utility
    # =================================================

    def get_business_value(
        self,
        fact_id
    ):

        return self.utility_map.get(
            fact_id,
            0.0
        )

    def calculate_utility(
        self,
        facts
    ):

        return sum(
            self.get_business_value(
                fact
            )
            for fact in facts
        )

    # =================================================
    # Effective Authorization
    # =================================================

    def get_authorization_context(
        self,
        user_id,
        as_of=None
    ):

        return (
            self.dynamic_authorization
            .get_effective_context(
                user_id=user_id,
                as_of=as_of
            )
        )

    # =================================================
    # 정보조합 권한 판단
    # =================================================

    def can_receive_combination(
        self,
        user_id,
        rule,
        as_of=None
    ):

        context = (
            self.get_authorization_context(
                user_id,
                as_of
            )
        )

        clearance_ok = (
            context[
                "effective_clearance"
            ]
            >= rule[
                "required_clearance"
            ]
        )

        allowed_missions = set(
            rule[
                "allowed_missions"
            ]
        )

        effective_missions = set(
            context[
                "effective_missions"
            ]
        )

        mission_ok = bool(
            allowed_missions
            & effective_missions
        )

        return (
            clearance_ok
            and mission_ok
        )

    # =================================================
    # 현재 위반 규칙
    # =================================================

    def find_violations(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None
    ):

        available_facts = (
            set(
                exposed_facts
            )
            | set(
                candidate_facts
            )
        )

        violations = []

        for rule in self.rules:

            required = set(
                rule[
                    "required_facts"
                ]
            )

            if not required.issubset(
                available_facts
            ):

                continue

            if self.can_receive_combination(
                user_id=user_id,
                rule=rule,
                as_of=as_of
            ):

                continue

            violations.append(
                rule
            )

        return violations

    # =================================================
    # 과거에 이미 발생한 위반
    # =================================================

    def find_preexisting_violations(
        self,
        user_id,
        exposed_facts,
        as_of=None
    ):

        exposed = set(
            exposed_facts
        )

        violations = []

        for rule in self.rules:

            required = set(
                rule[
                    "required_facts"
                ]
            )

            if not required.issubset(
                exposed
            ):

                continue

            if self.can_receive_combination(
                user_id=user_id,
                rule=rule,
                as_of=as_of
            ):

                continue

            violations.append(
                rule
            )

        return violations

    # =================================================
    # 제거 후 안전성
    # =================================================

    def is_safe_after_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        removed_facts,
        as_of=None
    ):

        removed_set = set(
            removed_facts
        )

        remaining_candidates = [
            fact
            for fact in candidate_facts
            if fact not in removed_set
        ]

        violations = (
            self.find_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                candidate_facts=
                    remaining_candidates,

                as_of=
                    as_of
            )
        )

        preexisting = {
            rule["rule_id"]
            for rule
            in self.find_preexisting_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                as_of=
                    as_of
            )
        }

        new_violations = [
            rule
            for rule in violations
            if (
                rule["rule_id"]
                not in preexisting
            )
        ]

        return (
            len(
                new_violations
            )
            == 0
        )

    # =================================================
    # 가중 최소 업무손실 제거
    # =================================================

    def find_minimum_loss_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None
    ):

        candidate_facts = list(
            dict.fromkeys(
                candidate_facts
            )
        )

        # 아무것도 제거하지 않아도 안전
        if self.is_safe_after_removal(
            user_id=
                user_id,

            exposed_facts=
                exposed_facts,

            candidate_facts=
                candidate_facts,

            removed_facts=
                [],

            as_of=
                as_of
        ):

            return []

        best_removal = None
        best_cost = float(
            "inf"
        )

        best_count = float(
            "inf"
        )

        number_of_candidates = len(
            candidate_facts
        )

        for size in range(
            1,
            number_of_candidates + 1
        ):

            for subset in combinations(
                candidate_facts,
                size
            ):

                removed = list(
                    subset
                )

                if not (
                    self.is_safe_after_removal(
                        user_id=
                            user_id,

                        exposed_facts=
                            exposed_facts,

                        candidate_facts=
                            candidate_facts,

                        removed_facts=
                            removed,

                        as_of=
                            as_of
                    )
                ):

                    continue

                loss = (
                    self.calculate_utility(
                        removed
                    )
                )

                should_replace = False

                # 1순위:
                # 총 업무가치 손실 최소
                if loss < best_cost:

                    should_replace = True

                # 2순위:
                # 동일 손실이면 제거 Fact 수 최소
                elif (
                    loss == best_cost
                    and len(
                        removed
                    ) < best_count
                ):

                    should_replace = True

                # 3순위:
                # 완전 동일 조건이면 결정론적 선택
                elif (
                    loss == best_cost
                    and len(
                        removed
                    ) == best_count
                    and best_removal
                    is not None
                    and tuple(
                        sorted(
                            removed
                        )
                    )
                    < tuple(
                        sorted(
                            best_removal
                        )
                    )
                ):

                    should_replace = True

                if should_replace:

                    best_removal = (
                        removed
                    )

                    best_cost = (
                        loss
                    )

                    best_count = len(
                        removed
                    )

        # candidate 전체를 지워도 신규 위반을
        # 해결할 수 없다면 현재 Turn에서
        # 해결 가능한 위반이 없는 것으로 본다.
        if best_removal is None:

            return []

        return best_removal

    # =================================================
    # Turn 전체 평가
    # =================================================

    def evaluate_turn(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None
    ):

        candidate_facts = list(
            dict.fromkeys(
                candidate_facts
            )
        )

        authorization_context = (
            self.get_authorization_context(
                user_id=
                    user_id,

                as_of=
                    as_of
            )
        )

        initial_violations = (
            self.find_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                candidate_facts=
                    candidate_facts,

                as_of=
                    as_of
            )
        )

        preexisting_violations = (
            self.find_preexisting_violations(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                as_of=
                    as_of
            )
        )

        removed_facts = (
            self.find_minimum_loss_removal(
                user_id=
                    user_id,

                exposed_facts=
                    exposed_facts,

                candidate_facts=
                    candidate_facts,

                as_of=
                    as_of
            )
        )

        removed_set = set(
            removed_facts
        )

        allowed_facts = [
            fact
            for fact in candidate_facts
            if fact not in removed_set
        ]

        candidate_utility = (
            self.calculate_utility(
                candidate_facts
            )
        )

        removed_utility = (
            self.calculate_utility(
                removed_facts
            )
        )

        retained_utility = (
            self.calculate_utility(
                allowed_facts
            )
        )

        if candidate_utility > 0:

            utility_retention_rate = (
                retained_utility
                / candidate_utility
            )

        else:

            utility_retention_rate = (
                1.0
            )

        return {
            "user_id":
                user_id,

            "as_of":
                (
                    str(as_of)
                    if as_of is not None
                    else None
                ),

            "authorization_context":
                authorization_context,

            "previously_exposed":
                list(
                    exposed_facts
                ),

            "candidate_facts":
                candidate_facts,

            "detected_rules":
                [
                    rule[
                        "rule_id"
                    ]
                    for rule
                    in initial_violations
                ],

            "preexisting_violations":
                [
                    rule[
                        "rule_id"
                    ]
                    for rule
                    in preexisting_violations
                ],

            "removed_facts":
                removed_facts,

            "allowed_facts":
                allowed_facts,

            "candidate_utility":
                candidate_utility,

            "removed_utility":
                removed_utility,

            "retained_utility":
                retained_utility,

            "utility_retention_rate":
                utility_retention_rate
        }


# =====================================================
# 실행 예제
# =====================================================

if __name__ == "__main__":

    controller = (
        IncrementalDisclosureController()
    )

    candidate_facts = [
        "OPS001-F2",
        "OPS002-F2"
    ]

    times = [
        (
            "BEFORE",
            "2026-09-29T08:30:00+09:00"
        ),
        (
            "ACTIVE",
            "2026-09-29T10:00:00+09:00"
        ),
        (
            "EXPIRED",
            "2026-09-29T12:30:00+09:00"
        )
    ]

    for label, as_of in times:

        print(
            "\n"
            + "=" * 60
        )

        print(
            label
        )

        result = (
            controller.evaluate_turn(
                user_id=
                    "U3",

                exposed_facts=
                    [],

                candidate_facts=
                    candidate_facts,

                as_of=
                    as_of
            )
        )

        print(
            "Authorization:",
            result[
                "authorization_context"
            ]
        )

        print(
            "Detected Rules:",
            result[
                "detected_rules"
            ]
        )

        print(
            "Removed Facts:",
            result[
                "removed_facts"
            ]
        )

        print(
            "Allowed Facts:",
            result[
                "allowed_facts"
            ]
        )
