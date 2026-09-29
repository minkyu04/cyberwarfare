import json
from itertools import combinations
from pathlib import Path


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
    이전 공개정보와 신규 공개정보의 결합으로 발생하는
    금지된 정보조합을 탐지하고,

    금지조합을 모두 차단하면서
    업무정보 손실량을 최소화하는 Fact 집합을 계산한다.
    """

    def __init__(self):

        self.users = load_json("users.json")
        self.rules = load_json("inference_rules.json")
        self.fact_utility = load_json(
            "fact_utility.json"
        )

        self.user_map = {
            user["user_id"]: user
            for user in self.users
        }

        self.utility_map = {
            item["fact_id"]:
                float(item["business_value"])
            for item in self.fact_utility
        }

        self.validate_utility_values()

    # =================================================
    # 초기 데이터 검증
    # =================================================

    def validate_utility_values(self):
        """
        inference rule에서 사용하는 모든 Fact에
        업무가치가 정의되어 있는지 확인한다.
        """

        required_fact_ids = set()

        for rule in self.rules:
            required_fact_ids.update(
                rule["required_facts"]
            )

        missing = (
            required_fact_ids
            - set(self.utility_map.keys())
        )

        if missing:
            raise ValueError(
                "업무가치가 정의되지 않은 Fact가 있습니다: "
                f"{sorted(missing)}"
            )

    # =================================================
    # 사용자 정보
    # =================================================

    def get_user(self, user_id):

        if user_id not in self.user_map:
            raise ValueError(
                f"Unknown user: {user_id}"
            )

        return self.user_map[user_id]

    # =================================================
    # Fact 업무가치
    # =================================================

    def get_business_value(self, fact_id):

        return self.utility_map.get(
            fact_id,
            0.0
        )

    def calculate_utility(self, facts):

        return sum(
            self.get_business_value(fact)
            for fact in facts
        )

    # =================================================
    # 정보조합 권한
    # =================================================

    def can_receive_combination(
        self,
        user,
        rule
    ):

        clearance_ok = (
            user["clearance"]
            >= rule["required_clearance"]
        )

        mission_ok = (
            user["mission"]
            in rule["allowed_missions"]
        )

        return (
            clearance_ok
            and mission_ok
        )

    # =================================================
    # 현재 위반 규칙 확인
    # =================================================

    def find_violations(
        self,
        user_id,
        exposed_facts,
        candidate_facts
    ):

        user = self.get_user(user_id)

        available_facts = (
            set(exposed_facts)
            | set(candidate_facts)
        )

        violations = []

        for rule in self.rules:

            required = set(
                rule["required_facts"]
            )

            if not required.issubset(
                available_facts
            ):
                continue

            if self.can_receive_combination(
                user,
                rule
            ):
                continue

            violations.append(rule)

        return violations

    # =================================================
    # 과거에 이미 발생한 위반
    # =================================================

    def find_preexisting_violations(
        self,
        user_id,
        exposed_facts
    ):

        user = self.get_user(user_id)

        exposed = set(exposed_facts)

        violations = []

        for rule in self.rules:

            required = set(
                rule["required_facts"]
            )

            if not required.issubset(
                exposed
            ):
                continue

            if self.can_receive_combination(
                user,
                rule
            ):
                continue

            violations.append(rule)

        return violations

    # =================================================
    # 특정 Fact 제거 후 안전성 검사
    # =================================================

    def is_safe_after_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        removed_facts
    ):

        removed_set = set(
            removed_facts
        )

        remaining_candidates = [
            fact
            for fact in candidate_facts
            if fact not in removed_set
        ]

        violations = self.find_violations(
            user_id=user_id,
            exposed_facts=exposed_facts,
            candidate_facts=remaining_candidates
        )

        preexisting = {
            rule["rule_id"]
            for rule
            in self.find_preexisting_violations(
                user_id,
                exposed_facts
            )
        }

        new_violations = [
            rule
            for rule in violations
            if rule["rule_id"]
            not in preexisting
        ]

        return len(new_violations) == 0

    # =================================================
    # 핵심:
    # 가중 최소 업무손실 제거 알고리즘
    # =================================================

    def find_minimum_loss_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts
    ):

        candidate_facts = list(
            dict.fromkeys(candidate_facts)
        )

        # 아무것도 제거하지 않아도 안전
        if self.is_safe_after_removal(
            user_id,
            exposed_facts,
            candidate_facts,
            []
        ):
            return []

        best_removal = None
        best_cost = float("inf")
        best_count = float("inf")

        number_of_candidates = len(
            candidate_facts
        )

        # 모든 가능한 제거조합 탐색
        for size in range(
            1,
            number_of_candidates + 1
        ):

            for subset in combinations(
                candidate_facts,
                size
            ):

                removed = list(subset)

                # 보안조건을 만족하지 않으면 제외
                if not self.is_safe_after_removal(
                    user_id,
                    exposed_facts,
                    candidate_facts,
                    removed
                ):
                    continue

                loss = self.calculate_utility(
                    removed
                )

                # -------------------------------------
                # 1순위: 업무손실 최소
                # 2순위: 제거 Fact 수 최소
                # 3순위: 결과 재현성을 위한 사전식 순서
                # -------------------------------------

                should_replace = False

                if loss < best_cost:
                    should_replace = True

                elif (
                    loss == best_cost
                    and len(removed) < best_count
                ):
                    should_replace = True

                elif (
                    loss == best_cost
                    and len(removed) == best_count
                    and best_removal is not None
                    and tuple(sorted(removed))
                    < tuple(sorted(best_removal))
                ):
                    should_replace = True

                if should_replace:

                    best_removal = removed

                    best_cost = loss

                    best_count = len(
                        removed
                    )

        if best_removal is None:

            # 현재 candidate를 전부 제거해도
            # 신규 위반을 해결할 수 없는 경우
            # 일반적으로 과거에 이미 위반이 완성된 경우
            return []

        return best_removal

    # =================================================
    # 한 Turn 평가
    # =================================================

    def evaluate_turn(
        self,
        user_id,
        exposed_facts,
        candidate_facts
    ):

        candidate_facts = list(
            dict.fromkeys(candidate_facts)
        )

        initial_violations = (
            self.find_violations(
                user_id,
                exposed_facts,
                candidate_facts
            )
        )

        preexisting_violations = (
            self.find_preexisting_violations(
                user_id,
                exposed_facts
            )
        )

        removed_facts = (
            self.find_minimum_loss_removal(
                user_id,
                exposed_facts,
                candidate_facts
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

            utility_retention_rate = 1.0

        return {
            "user_id": user_id,

            "previously_exposed":
                list(exposed_facts),

            "candidate_facts":
                candidate_facts,

            "detected_rules": [
                rule["rule_id"]
                for rule in initial_violations
            ],

            "preexisting_violations": [
                rule["rule_id"]
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

    print(
        "\n=== Weighted Minimum Loss Test ==="
    )

    result = controller.evaluate_turn(
        user_id="U3",
        exposed_facts=[],
        candidate_facts=[
            "OPS001-F2",
            "OPS002-F2",
            "OPS002-F3"
        ]
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n=== Multi-turn Test ==="
    )

    result = controller.evaluate_turn(
        user_id="U3",
        exposed_facts=[
            "OPS001-F2"
        ],
        candidate_facts=[
            "OPS002-F2",
            "OPS002-F3"
        ]
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\n=== Authorized User Test ==="
    )

    result = controller.evaluate_turn(
        user_id="U4",
        exposed_facts=[],
        candidate_facts=[
            "LOG001-F3",
            "LOG002-F3"
        ]
    )

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )
