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
    과거 공개정보와 현재 답변 후보의 조합을 분석한다.

    목표
    --------------------------------------------------
    1. 금지 정보조합의 신규 완성을 방지한다.
    2. 과거에 이미 발생한 위반을 현재 Turn의 신규 위반과 구분한다.
    3. 현재 응답 자체가 금지조합을 다시 완성하는 것도 차단한다.
    4. 금지조합을 해소하면서 업무정보 손실을 최소화한다.
    5. 동적 임시권한을 Turn 단위로 재평가한다.
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
                    item["business_value"]
                )
            for item in self.fact_utility
        }

        self.dynamic_authorization = (
            DynamicAuthorizationManager()
        )

        self.validate_utility_values()

    # =================================================
    # Utility 데이터 검증
    # =================================================

    def validate_utility_values(self):

        required_fact_ids = set()

        for rule in self.rules:

            required_fact_ids.update(
                rule["required_facts"]
            )

        missing = (
            required_fact_ids
            - set(
                self.utility_map.keys()
            )
        )

        if missing:

            raise ValueError(
                "업무가치가 정의되지 않은 Fact가 있습니다: "
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
    # 현재 Effective Authorization
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
    # History + Candidate 전체에서 위반 탐지
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
    # History만으로 이미 존재하던 위반
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
    # 현재 응답 때문에 처리해야 하는 위반
    # =================================================

    def find_current_response_violations(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        as_of=None
    ):
        """
        다음 두 경우를 현재 응답의 위반으로 본다.

        A.
        과거에는 완성되지 않았으나,
        이번 Candidate가 추가되면서 금지조합이 완성됨.

        B.
        과거 History에 이미 금지조합이 존재하더라도,
        현재 응답 자체가 해당 금지조합 전체를 다시 제공함.

        반대로,
        History에 이미 존재하던 금지조합과 관계없는
        개별 Fact 하나를 정상적으로 제공하는 것은
        현재 응답 위반으로 보지 않는다.
        """

        candidate_set = set(
            candidate_facts
        )

        combined_violations = (
            self.find_violations(
                user_id=user_id,
                exposed_facts=exposed_facts,
                candidate_facts=candidate_facts,
                as_of=as_of
            )
        )

        preexisting_ids = {
            rule["rule_id"]
            for rule
            in self.find_preexisting_violations(
                user_id=user_id,
                exposed_facts=exposed_facts,
                as_of=as_of
            )
        }

        current_violations = []

        for rule in combined_violations:

            rule_id = rule[
                "rule_id"
            ]

            required = set(
                rule[
                    "required_facts"
                ]
            )

            # -----------------------------------------
            # 신규로 완성된 위반
            # -----------------------------------------

            if rule_id not in preexisting_ids:

                current_violations.append(
                    rule
                )

                continue

            # -----------------------------------------
            # 과거에도 위반이 있었지만
            # 현재 답변 자체가 금지조합 전체를
            # 다시 제공하려는 경우
            # -----------------------------------------

            if required.issubset(
                candidate_set
            ):

                current_violations.append(
                    rule
                )

        return current_violations

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

        current_violations = (
            self.find_current_response_violations(
                user_id=user_id,
                exposed_facts=exposed_facts,
                candidate_facts=
                    remaining_candidates,
                as_of=as_of
            )
        )

        return (
            len(
                current_violations
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

        # 제거하지 않아도 안전한 경우
        if self.is_safe_after_removal(
            user_id=user_id,
            exposed_facts=exposed_facts,
            candidate_facts=candidate_facts,
            removed_facts=[],
            as_of=as_of
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

                if not self.is_safe_after_removal(
                    user_id=user_id,
                    exposed_facts=
                        exposed_facts,
                    candidate_facts=
                        candidate_facts,
                    removed_facts=
                        removed,
                    as_of=
                        as_of
                ):

                    continue

                loss = (
                    self.calculate_utility(
                        removed
                    )
                )

                should_replace = False

                # -------------------------------------
                # 1순위
                # 업무가치 손실 최소
                # -------------------------------------

                if loss < best_cost:

                    should_replace = True

                # -------------------------------------
                # 2순위
                # 동일 손실이면 제거 Fact 수 최소
                # -------------------------------------

                elif (
                    loss == best_cost
                    and len(
                        removed
                    ) < best_count
                ):

                    should_replace = True

                # -------------------------------------
                # 3순위
                # 동일 조건이면 결정론적 순서
                # -------------------------------------

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
                user_id=user_id,
                as_of=as_of
            )
        )

        detected_violations = (
            self.find_violations(
                user_id=user_id,
                exposed_facts=
                    exposed_facts,
                candidate_facts=
                    candidate_facts,
                as_of=as_of
            )
        )

        preexisting_violations = (
            self.find_preexisting_violations(
                user_id=user_id,
                exposed_facts=
                    exposed_facts,
                as_of=as_of
            )
        )

        current_response_violations = (
            self.find_current_response_violations(
                user_id=user_id,
                exposed_facts=
                    exposed_facts,
                candidate_facts=
                    candidate_facts,
                as_of=as_of
            )
        )

        removed_facts = (
            self.find_minimum_loss_removal(
                user_id=user_id,
                exposed_facts=
                    exposed_facts,
                candidate_facts=
                    candidate_facts,
                as_of=as_of
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
                    in detected_violations
                ],

            "preexisting_violations":
                [
                    rule[
                        "rule_id"
                    ]
                    for rule
                    in preexisting_violations
                ],

            "current_response_violations":
                [
                    rule[
                        "rule_id"
                    ]
                    for rule
                    in current_response_violations
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
