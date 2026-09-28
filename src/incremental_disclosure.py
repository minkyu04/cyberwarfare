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
    이전에 공개된 Fact와 이번 응답에서 공개하려는 Fact를 결합하여
    금지된 정보조합이 형성되는지 검사하고,
    해당 조합을 차단하기 위한 최소 제거 집합을 계산한다.
    """

    def __init__(self):

        self.users = load_json("users.json")
        self.rules = load_json("inference_rules.json")

        self.user_map = {
            user["user_id"]: user
            for user in self.users
        }

    # -------------------------------------------------
    # 사용자 정보
    # -------------------------------------------------

    def get_user(self, user_id):

        if user_id not in self.user_map:
            raise ValueError(
                f"Unknown user: {user_id}"
            )

        return self.user_map[user_id]

    # -------------------------------------------------
    # 사용자가 해당 정보조합을 받을 권한이 있는지
    # -------------------------------------------------

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

    # -------------------------------------------------
    # 현재 상태에서 위반되는 규칙 탐지
    # -------------------------------------------------

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

            # 해당 조합 자체가 완성되지 않았으면 문제 없음
            if not required.issubset(
                available_facts
            ):
                continue

            # 사용자가 해당 조합을 받을 권한이 있으면 문제 없음
            if self.can_receive_combination(
                user,
                rule
            ):
                continue

            violations.append(rule)

        return violations

    # -------------------------------------------------
    # 이미 이전 턴에서 발생한 위반인지 검사
    # -------------------------------------------------

    def find_preexisting_violations(
        self,
        user_id,
        exposed_facts
    ):

        user = self.get_user(user_id)
        history = set(exposed_facts)

        violations = []

        for rule in self.rules:

            required = set(
                rule["required_facts"]
            )

            if not required.issubset(history):
                continue

            if self.can_receive_combination(
                user,
                rule
            ):
                continue

            violations.append(rule)

        return violations

    # -------------------------------------------------
    # 제거 후에도 금지 조합이 남아있는지 검사
    # -------------------------------------------------

    def is_safe_after_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts,
        removed_facts
    ):

        remaining_candidates = [
            fact
            for fact in candidate_facts
            if fact not in removed_facts
        ]

        violations = self.find_violations(
            user_id=user_id,
            exposed_facts=exposed_facts,
            candidate_facts=remaining_candidates
        )

        # 이전 대화만으로 이미 발생한 위반은
        # 현재 답변에서 해결할 수 없으므로 제외하고 판단
        history_only_violations = {
            rule["rule_id"]
            for rule in self.find_preexisting_violations(
                user_id,
                exposed_facts
            )
        }

        new_violations = [
            rule
            for rule in violations
            if rule["rule_id"]
            not in history_only_violations
        ]

        return len(new_violations) == 0

    # -------------------------------------------------
    # 최소 제거 Fact 계산
    # -------------------------------------------------

    def find_minimum_removal(
        self,
        user_id,
        exposed_facts,
        candidate_facts
    ):

        candidate_facts = list(
            dict.fromkeys(candidate_facts)
        )

        # 아무것도 제거하지 않아도 안전한지 먼저 확인
        if self.is_safe_after_removal(
            user_id,
            exposed_facts,
            candidate_facts,
            []
        ):
            return []

        # 제거 개수 1개 → 2개 → ... 순서로 탐색
        # 첫 번째로 안전해지는 조합이 최소 제거 집합
        for size in range(
            1,
            len(candidate_facts) + 1
        ):

            for subset in combinations(
                candidate_facts,
                size
            ):

                removed = list(subset)

                if self.is_safe_after_removal(
                    user_id,
                    exposed_facts,
                    candidate_facts,
                    removed
                ):
                    return removed

        # 모든 후보를 제거해도 해결할 수 없다면
        # 이전 턴에서 이미 위반된 경우일 가능성이 높음
        return candidate_facts

    # -------------------------------------------------
    # 한 턴 전체 평가
    # -------------------------------------------------

    def evaluate_turn(
        self,
        user_id,
        exposed_facts,
        candidate_facts
    ):

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

        removal = self.find_minimum_removal(
            user_id,
            exposed_facts,
            candidate_facts
        )

        allowed_facts = [
            fact
            for fact in candidate_facts
            if fact not in removal
        ]

        return {
            "user_id": user_id,

            "previously_exposed":
                list(exposed_facts),

            "candidate_facts":
                list(candidate_facts),

            "detected_rules": [
                rule["rule_id"]
                for rule in initial_violations
            ],

            "preexisting_violations": [
                rule["rule_id"]
                for rule
                in preexisting_violations
            ],

            "removed_facts": removal,

            "allowed_facts":
                allowed_facts
        }


# =====================================================
# 간단한 실행 테스트
# =====================================================

if __name__ == "__main__":

    controller = (
        IncrementalDisclosureController()
    )

    print(
        "\n=== Test 1: Multi-turn Leakage ==="
    )

    result = controller.evaluate_turn(
        user_id="U3",
        exposed_facts=[
            "OPS001-F2"
        ],
        candidate_facts=[
            "OPS002-F2"
        ]
    )

    print(result)


    print(
        "\n=== Test 2: Minimum Removal ==="
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

    print(result)


    print(
        "\n=== Test 3: Authorized Combination ==="
    )

    result = controller.evaluate_turn(
        user_id="U4",
        exposed_facts=[],
        candidate_facts=[
            "LOG001-F3",
            "LOG002-F3"
        ]
    )

    print(result)


    print(
        "\n=== Test 4: Unauthorized Combination ==="
    )

    result = controller.evaluate_turn(
        user_id="U4",
        exposed_facts=[],
        candidate_facts=[
            "VUL001-F3",
            "VUL002-F3"
        ]
    )

    print(result)
